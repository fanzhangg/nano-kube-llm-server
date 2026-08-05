/*
Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controller

import (
	"context"
	"fmt"
	"strconv"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	appsv1ac "k8s.io/client-go/applyconfigurations/apps/v1"
	corev1ac "k8s.io/client-go/applyconfigurations/core/v1"
	metav1ac "k8s.io/client-go/applyconfigurations/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	servingv1alpha1 "github.com/fanzhangg/nano-kube-llm-server/api/v1alpha1"
)

// gpuResourceName is the extended resource advertised by the NVIDIA device
// plugin. Scheduling a pod that requests it fails unless that plugin is running
// on the cluster, which is why spec.gpus defaults to 0.
const gpuResourceName corev1.ResourceName = "nvidia.com/gpu"

const (
	// instanceLabelKey ties an owned object back to the ModelServer that created
	// it. It is both a metadata label and the Service's Pod selector, so it must
	// stay identical in all three places -- hence the constant.
	instanceLabelKey = "serving.fanzhangg.dev/instance"

	// appLabelKey / appLabelValue mark everything this operator manages,
	// independent of which ModelServer it belongs to.
	appLabelKey   = "app"
	appLabelValue = "modelserver"
)

// ModelServerReconciler reconciles a ModelServer object
type ModelServerReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=serving.fanzhangg.dev,resources=modelservers,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=serving.fanzhangg.dev,resources=modelservers/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=serving.fanzhangg.dev,resources=modelservers/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.

// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.24.1/pkg/reconcile
func (r *ModelServerReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Get the ModelServer object
	var ms servingv1alpha1.ModelServer
	if err := r.Get(ctx, req.NamespacedName, &ms); err != nil {
		// Ignore not found error. The resource is being deleted, do nothing
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	// Reconcile the deployment to the expected state
	if err := r.reconcileDeployment(ctx, &ms); err != nil {
		logger.Error(err, "Failed to reconcile deployment")
		return ctrl.Result{}, err
	}

	// Reconcile the service to the expected state
	if err := r.reconcileService(ctx, &ms); err != nil {
		logger.Error(err, "Failed to reconcile service")
		return ctrl.Result{}, err
	}

	// Update the status
	if err := r.updateStatus(ctx, &ms); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *ModelServerReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&servingv1alpha1.ModelServer{}).
		Owns(&appsv1.Deployment{}). // Change in deployment will trigger reconcile
		Owns(&corev1.Service{}).
		Complete(r)
}

func (r *ModelServerReconciler) updateStatus(ctx context.Context, ms *servingv1alpha1.ModelServer) error {
	var deploy appsv1.Deployment
	err := r.Get(ctx, types.NamespacedName{Name: ms.Name, Namespace: ms.Namespace}, &deploy)
	if err != nil && !errors.IsNotFound(err) {
		return err
	}
	found := err == nil

	var ready, total int32

	if found {
		ready = deploy.Status.ReadyReplicas
		total = deploy.Status.Replicas
	}

	available := metav1.ConditionFalse
	availableReason := "DeploymentNotFound"

	if found {
		if ready >= ms.Spec.Replicas {
			available = metav1.ConditionTrue
			availableReason = "AllReplicasReady"
		} else {
			availableReason = "ReplicasNotReady"
		}
	}

	changed := meta.SetStatusCondition(&ms.Status.Conditions, metav1.Condition{
		Type:    servingv1alpha1.ConditionTypeAvailable,
		Status:  available,
		Reason:  availableReason,
		Message: fmt.Sprintf("%d/%d replicas ready", ready, ms.Spec.Replicas),
	})

	progressing := metav1.ConditionTrue
	progressingReason := "Reconciling"
	if available == metav1.ConditionTrue {
		progressing = metav1.ConditionFalse
		progressingReason = "AllReplicasReady"
	}

	if meta.SetStatusCondition(&ms.Status.Conditions, metav1.Condition{
		Type:   servingv1alpha1.ConditionTypeProgressing,
		Status: progressing,
		Reason: progressingReason,
	}) {
		changed = true
	}

	phase := "Loading"
	switch {
	case !found || total == 0:
		phase = "Pending"
	case available == metav1.ConditionTrue:
		phase = "Ready"
	}
	if ms.Status.Phase != phase {
		ms.Status.Phase = phase
		changed = true
	}
	if ms.Status.ReadyReplicas != ready {
		ms.Status.ReadyReplicas = ready
		changed = true
	}
	if ms.Status.ObservedGeneration != ms.Generation {
		ms.Status.ObservedGeneration = ms.Generation
		changed = true
	}

	if !changed {
		return nil
	}

	return r.Status().Update(ctx, ms)
}

func (r *ModelServerReconciler) reconcileDeployment(ctx context.Context, ms *servingv1alpha1.ModelServer) error {
	desired := r.buildDeployment(ms)

	return r.Apply(ctx, desired, client.FieldOwner("modelserver-controller"), client.ForceOwnership)
}

// ownerRef builds the controller reference stamped on every object this
// reconciler owns, so deleting the ModelServer garbage-collects them.
//
// ctrl.SetControllerReference cannot be used here: it takes a metav1.Object, and
// an apply configuration is not one. Hand-building it is the documented way to
// express ownership when using server-side apply.
func ownerRef(ms *servingv1alpha1.ModelServer) *metav1ac.OwnerReferenceApplyConfiguration {
	return metav1ac.OwnerReference().
		WithAPIVersion(servingv1alpha1.GroupVersion.String()).
		WithKind("ModelServer").
		WithName(ms.Name).
		WithUID(ms.UID).
		WithController(true).
		WithBlockOwnerDeletion(true)
}

// instanceLabels identify everything belonging to one ModelServer. They are
// stamped on the objects themselves, not only on the Pod template, so that
// `kubectl get deploy,svc -l app=modelserver` finds what this operator manages.
// Pods inherit them from the template; the Deployment and Service do not, which
// is why they have to be set explicitly.
func instanceLabels(ms *servingv1alpha1.ModelServer) map[string]string {
	return map[string]string{
		appLabelKey:      appLabelValue,
		instanceLabelKey: ms.Name,
	}
}

func (r *ModelServerReconciler) buildDeployment(ms *servingv1alpha1.ModelServer) *appsv1ac.DeploymentApplyConfiguration {
	labels := instanceLabels(ms)

	container := corev1ac.Container().
		WithName("server").
		WithImage(ms.Spec.Image).
		WithImagePullPolicy(corev1.PullIfNotPresent).
		WithPorts(corev1ac.ContainerPort().WithContainerPort(8000)).
		// MODEL_ID is what makes the server load real weights; MODEL_NAME is only
		// the reported label. Both come from spec.model: configuring a model is
		// what selects the engine, so there is no separate switch to keep in sync.
		WithEnv(
			corev1ac.EnvVar().WithName("MODEL_ID").WithValue(ms.Spec.Model),
			corev1ac.EnvVar().WithName("MODEL_NAME").WithValue(ms.Spec.Model),
			corev1ac.EnvVar().WithName("MAX_BATCH_SIZE").WithValue(strconv.Itoa(int(ms.Spec.MaxBatchSize))),
		).
		// Three probes, one endpoint, three different questions. /health answers
		// 503 while the model is loading AND 503 once the inference loop has
		// died; which of those a 503 means depends only on WHEN it is asked,
		// and that is exactly what the probe split encodes.
		//
		// The startup probe is what makes a liveness probe safe here. Loading
		// weights takes tens of seconds for a 0.6B and minutes for anything
		// larger, and a liveness probe on its own cannot tell "still loading"
		// from "wedged" -- it would restart the pod mid-load, then restart the
		// restart, which is the classic way to turn a slow start into a
		// CrashLoopBackOff. Kubernetes disables liveness and readiness until the
		// startup probe first succeeds, so the whole load window is protected.
		//
		// 120 x 5s = a 10 minute budget. A pod that cannot load in ten minutes
		// now crash-loops instead of sitting in Loading forever with nothing in
		// status to explain it -- a wrong spec.model used to fail exactly that
		// silently.
		WithStartupProbe(corev1ac.Probe().
			WithHTTPGet(corev1ac.HTTPGetAction().
				WithPath("/health").
				WithPort(intstr.FromInt(8000))).
			WithPeriodSeconds(5).
			WithFailureThreshold(120)).
		// After startup succeeds, `loaded` is true for the life of the process
		// and never flips back -- so from here on a 503 can only mean the run
		// loop is gone. That is unrecoverable in-process (see run_loop: after a
		// CUDA OOM the allocator state is unknown), and a restart is the whole
		// remedy run_loop's docstring promises. ~30s to notice.
		WithLivenessProbe(corev1ac.Probe().
			WithHTTPGet(corev1ac.HTTPGetAction().
				WithPath("/health").
				WithPort(intstr.FromInt(8000))).
			WithPeriodSeconds(10).
			WithFailureThreshold(3)).
		// No InitialDelaySeconds: the startup probe already gates this one, so a
		// delay here would only postpone the first check after loading finished.
		WithReadinessProbe(corev1ac.Probe().
			WithHTTPGet(corev1ac.HTTPGetAction().
				WithPath("/health").
				WithPort(intstr.FromInt(8000))).
			WithPeriodSeconds(3))

	// Only set resources when GPUs are actually requested: an empty ResourceList
	// on a CPU-only cluster is noise, and nvidia.com/gpu is meaningless without
	// the NVIDIA device plugin installed.
	if ms.Spec.GPUs > 0 {
		container = container.WithResources(corev1ac.ResourceRequirements().
			WithLimits(corev1.ResourceList{
				gpuResourceName: *resource.NewQuantity(int64(ms.Spec.GPUs), resource.DecimalSI),
			}))
	}

	// spec.selector is immutable once the Deployment exists, so it keeps matching
	// on the same two labels it always has. WithLabels only touches metadata,
	// which is free to change.
	return appsv1ac.Deployment(ms.Name, ms.Namespace).
		WithLabels(labels).
		WithOwnerReferences(ownerRef(ms)).
		WithSpec(appsv1ac.DeploymentSpec().
			WithReplicas(ms.Spec.Replicas).
			WithSelector(metav1ac.LabelSelector().WithMatchLabels(labels)).
			WithTemplate(corev1ac.PodTemplateSpec().
				WithLabels(labels).
				WithSpec(corev1ac.PodSpec().
					WithContainers(container))))
}

// reconcileService applies the Service, matching reconcileDeployment.
//
// The previous version was create-if-missing: once the Service existed it
// returned nil without comparing anything, so edits to the port or the selector
// were never corrected. Apply fixes that and removes the Get/IsNotFound/Create
// branching, since apply is create-or-update by nature.
//
// Not setting spec.clusterIP is deliberate rather than an omission. It is
// assigned by the API server and immutable, so a read-modify-write reconcile has
// to copy it off the existing object or every update is rejected with
// "spec.clusterIP: Invalid value: \"\": field is immutable". Under server-side
// apply the controller simply never claims the field, and the API server keeps
// its value. Same reasoning covers nodePort and ipFamilies if the type ever
// changes: list only what this controller means to own.
//
// Known limit, verified by the drift tests: spec.ports is a list-map keyed on
// (port, protocol), so apply corrects drift WITHIN the {port: 8000} entry it
// owns (targetPort, name, ...) but cannot delete an entry a different field
// manager added. Someone editing the port number to 9999 leaves a Service with
// both ports -- and since a multi-port Service requires names, every subsequent
// apply then fails validation with `spec.ports[1].name: Required value`, so the
// controller wedges until a human removes the stray port. ForceOwnership does
// not help: it resolves conflicts over the SAME field, and these are different
// list entries. Fixing it properly means owning the list atomically, which SSA
// does not offer for built-in types.
func (r *ModelServerReconciler) reconcileService(ctx context.Context, ms *servingv1alpha1.ModelServer) error {
	desired := corev1ac.Service(ms.Name, ms.Namespace).
		WithLabels(instanceLabels(ms)).
		WithOwnerReferences(ownerRef(ms)).
		WithSpec(corev1ac.ServiceSpec().
			// The selector stays on the instance label alone: it has to match Pods,
			// and narrowing to one ModelServer is the whole job. The metadata labels
			// above are for humans and kubectl, and are a separate concern.
			WithSelector(map[string]string{
				instanceLabelKey: ms.Name,
			}).
			WithPorts(corev1ac.ServicePort().
				WithPort(8000).
				WithTargetPort(intstr.FromInt(8000))))

	return r.Apply(ctx, desired, client.FieldOwner("modelserver-controller"), client.ForceOwnership)
}
