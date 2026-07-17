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

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	servingv1alpha1 "github.com/fanzhangg/nano-kube-llm-server/api/v1alpha1"
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
	return nil
}

func (r *ModelServerReconciler) reconcileDeployment(ctx context.Context, ms *servingv1alpha1.ModelServer) error {
	desired := r.buildDeployment(ms)

	if err := ctrl.SetControllerReference(ms, desired, r.Scheme); err != nil {
		return err
	}

	var existing appsv1.Deployment
	err := r.Get(ctx, types.NamespacedName{Name: desired.Name, Namespace: desired.Namespace}, &existing)

	// Create the model server if it does not exist
	if errors.IsNotFound(err) {
		return r.Create(ctx, desired)
	}

	if err != nil {
		return err
	}

	existing.Spec.Replicas = desired.Spec.Replicas
	existing.Spec.Template.Spec.Containers = desired.Spec.Template.Spec.Containers

	return r.Update(ctx, &existing)
}

func (r *ModelServerReconciler) buildDeployment(ms *servingv1alpha1.ModelServer) *appsv1.Deployment {
	labels := map[string]string{
		"app":                            "modelserver",
		"serving.fanzhangg.dev/instance": ms.Name,
	}
	replicas := ms.Spec.Replicas

	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ms.Name,
			Namespace: ms.Namespace,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: labels},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: labels},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "server",
						Image: ms.Spec.Image,
						Ports: []corev1.ContainerPort{{ContainerPort: 8000}},
						Env: []corev1.EnvVar{{
							Name:  "MODEL_NAME",
							Value: ms.Spec.Model,
						}},

						ReadinessProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{
								HTTPGet: &corev1.HTTPGetAction{
									Path: "/health",
									Port: intstr.FromInt(8000),
								},
							},
							InitialDelaySeconds: 5,
							PeriodSeconds:       3,
						},
					}},
				},
			},
		},
	}
}

func (r *ModelServerReconciler) reconcileService(ctx context.Context, ms *servingv1alpha1.ModelServer) error {
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ms.Name,
			Namespace: ms.Namespace,
		},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{
				"serving.fanzhangg.dev/instance": ms.Name,
			},
			Ports: []corev1.ServicePort{{
				Port:       8000,
				TargetPort: intstr.FromInt(8000),
			}},
		},
	}

	if err := ctrl.SetControllerReference(ms, svc, r.Scheme); err != nil {
		return err
	}

	var existing corev1.Service
	err := r.Get(ctx, types.NamespacedName{Name: svc.Name, Namespace: svc.Namespace}, &existing)
	if errors.IsNotFound(err) {
		return r.Create(ctx, svc)
	}
	if err != nil {
		return err
	}

	return nil
}
