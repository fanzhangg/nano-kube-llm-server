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

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	servingv1alpha1 "github.com/fanzhangg/nano-kube-llm-server/api/v1alpha1"
)

var _ = Describe("ModelServer Controller", func() {
	Context("When reconciling a resource", func() {
		const (
			resourceName      = "test-resource"
			resourceNamespace = "default"
			resourceModel     = "test-model"
		)

		ctx := context.Background()

		typeNamespacedName := types.NamespacedName{
			Name:      resourceName,
			Namespace: resourceNamespace,
		}
		modelserver := &servingv1alpha1.ModelServer{}

		BeforeEach(func() {
			By("creating the custom resource for the Kind ModelServer")
			err := k8sClient.Get(ctx, typeNamespacedName, modelserver)
			if err != nil && errors.IsNotFound(err) {
				resource := &servingv1alpha1.ModelServer{
					ObjectMeta: metav1.ObjectMeta{
						Name:      resourceName,
						Namespace: resourceNamespace,
					},
					Spec: servingv1alpha1.ModelServerSpec{
						Model: resourceModel,
					},
				}
				Expect(k8sClient.Create(ctx, resource)).To(Succeed())
			}
		})

		AfterEach(func() {
			// TODO(user): Cleanup logic after each test, like removing the resource instance.
			resource := &servingv1alpha1.ModelServer{}
			err := k8sClient.Get(ctx, typeNamespacedName, resource)
			Expect(err).NotTo(HaveOccurred())

			By("Cleanup the specific resource instance ModelServer")
			Expect(k8sClient.Delete(ctx, resource)).To(Succeed())
		})
		It("should successfully reconcile the resource", func() {
			By("Reconciling the created resource")
			controllerReconciler := &ModelServerReconciler{
				Client: k8sClient,
				Scheme: k8sClient.Scheme(),
			}

			_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())
			// TODO(user): Add more specific assertions depending on your controller's reconciliation logic.
			// Example: If you expect a certain status condition after reconciliation, verify it here.
		})
	})

	Context("When reporting status", func() {
		const (
			resourceName      = "status-test-resource"
			resourceNamespace = "default"
		)

		ctx := context.Background()
		typeNamespacedName := types.NamespacedName{Name: resourceName, Namespace: resourceNamespace}
		var reconciler *ModelServerReconciler

		BeforeEach(func() {
			resource := &servingv1alpha1.ModelServer{
				ObjectMeta: metav1.ObjectMeta{
					Name:      resourceName,
					Namespace: resourceNamespace,
				},
				Spec: servingv1alpha1.ModelServerSpec{
					Model:    "status-test-model",
					Replicas: 2,
				},
			}
			Expect(k8sClient.Create(ctx, resource)).To(Succeed())

			reconciler = &ModelServerReconciler{Client: k8sClient, Scheme: k8sClient.Scheme()}
		})

		AfterEach(func() {
			resource := &servingv1alpha1.ModelServer{}
			Expect(k8sClient.Get(ctx, typeNamespacedName, resource)).To(Succeed())
			Expect(k8sClient.Delete(ctx, resource)).To(Succeed())
		})

		// envtest only runs kube-apiserver + etcd -- there is no Deployment
		// controller or kubelet to report real Pod readiness into
		// Deployment.status. Patch it directly to simulate what a real cluster
		// would report at each stage (0 pods up, some ready, all ready).
		setDeploymentStatus := func(replicas, ready int32) {
			var deploy appsv1.Deployment
			Expect(k8sClient.Get(ctx, typeNamespacedName, &deploy)).To(Succeed())
			deploy.Status.Replicas = replicas
			deploy.Status.ReadyReplicas = ready
			Expect(k8sClient.Status().Update(ctx, &deploy)).To(Succeed())
		}

		doReconcile := func() servingv1alpha1.ModelServer {
			_, err := reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: typeNamespacedName})
			Expect(err).NotTo(HaveOccurred())
			var ms servingv1alpha1.ModelServer
			Expect(k8sClient.Get(ctx, typeNamespacedName, &ms)).To(Succeed())
			return ms
		}

		condition := func(ms servingv1alpha1.ModelServer, condType string) metav1.Condition {
			for _, c := range ms.Status.Conditions {
				if c.Type == condType {
					return c
				}
			}
			Fail(fmt.Sprintf("condition %q not found in %#v", condType, ms.Status.Conditions))
			return metav1.Condition{}
		}

		It("reports Pending right after creating the Deployment", func() {
			// First reconcile creates the Deployment; its status is still
			// zero-valued since nothing has reported readiness yet.
			ms := doReconcile()
			Expect(ms.Status.Phase).To(Equal("Pending"))
			Expect(condition(ms, servingv1alpha1.ConditionTypeAvailable).Status).To(Equal(metav1.ConditionFalse))
			Expect(condition(ms, servingv1alpha1.ConditionTypeAvailable).Reason).To(Equal("ReplicasNotReady"))
			Expect(condition(ms, servingv1alpha1.ConditionTypeProgressing).Status).To(Equal(metav1.ConditionTrue))
		})

		It("transitions Pending -> Loading -> Ready as replicas become ready", func() {
			doReconcile() // creates the Deployment

			setDeploymentStatus(2, 0) // pods scheduled, none ready yet
			ms := doReconcile()
			Expect(ms.Status.Phase).To(Equal("Loading"))
			Expect(ms.Status.ReadyReplicas).To(Equal(int32(0)))

			setDeploymentStatus(2, 1) // one of two ready
			ms = doReconcile()
			Expect(ms.Status.Phase).To(Equal("Loading"))
			Expect(ms.Status.ReadyReplicas).To(Equal(int32(1)))
			Expect(condition(ms, servingv1alpha1.ConditionTypeAvailable).Status).To(Equal(metav1.ConditionFalse))

			setDeploymentStatus(2, 2) // fully ready
			ms = doReconcile()
			Expect(ms.Status.Phase).To(Equal("Ready"))
			Expect(ms.Status.ReadyReplicas).To(Equal(int32(2)))
			Expect(condition(ms, servingv1alpha1.ConditionTypeAvailable).Status).To(Equal(metav1.ConditionTrue))
			Expect(condition(ms, servingv1alpha1.ConditionTypeAvailable).Reason).To(Equal("AllReplicasReady"))
			Expect(condition(ms, servingv1alpha1.ConditionTypeProgressing).Status).To(Equal(metav1.ConditionFalse))
			Expect(condition(ms, servingv1alpha1.ConditionTypeProgressing).Reason).To(Equal("AllReplicasReady"))
		})

		It("tracks ObservedGeneration", func() {
			ms := doReconcile()
			Expect(ms.Status.ObservedGeneration).To(Equal(ms.Generation))
		})
	})

	// The Day 4 exercise, as a test: someone bypasses the controller and edits an
	// owned object directly. Before reconcileService used server-side apply it was
	// create-if-missing -- it returned nil the moment the Service existed, so none
	// of these would have been corrected.
	Context("When an owned object is edited out of band", func() {
		const (
			resourceName      = "drift-test-resource"
			resourceNamespace = "default"
		)

		ctx := context.Background()
		typeNamespacedName := types.NamespacedName{Name: resourceName, Namespace: resourceNamespace}
		var reconciler *ModelServerReconciler

		reconcileOnce := func() {
			_, err := reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: typeNamespacedName})
			Expect(err).NotTo(HaveOccurred())
		}

		getService := func() corev1.Service {
			var svc corev1.Service
			Expect(k8sClient.Get(ctx, typeNamespacedName, &svc)).To(Succeed())
			return svc
		}

		BeforeEach(func() {
			resource := &servingv1alpha1.ModelServer{
				ObjectMeta: metav1.ObjectMeta{Name: resourceName, Namespace: resourceNamespace},
				Spec: servingv1alpha1.ModelServerSpec{
					Model:    "drift-test-model",
					Replicas: 1,
				},
			}
			Expect(k8sClient.Create(ctx, resource)).To(Succeed())
			reconciler = &ModelServerReconciler{Client: k8sClient, Scheme: k8sClient.Scheme()}
			reconcileOnce() // creates the Deployment and the Service
		})

		AfterEach(func() {
			resource := &servingv1alpha1.ModelServer{}
			Expect(k8sClient.Get(ctx, typeNamespacedName, resource)).To(Succeed())
			Expect(k8sClient.Delete(ctx, resource)).To(Succeed())

			// envtest runs kube-apiserver + etcd only -- there is no
			// garbage collector, so owner references do NOT cascade here.
			// Without deleting the owned objects explicitly, the next spec
			// reconciles on top of this spec's drifted Service.
			Expect(client.IgnoreNotFound(
				k8sClient.Delete(ctx, &corev1.Service{ObjectMeta: metav1.ObjectMeta{
					Name: resourceName, Namespace: resourceNamespace}}))).To(Succeed())
			Expect(client.IgnoreNotFound(
				k8sClient.Delete(ctx, &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{
					Name: resourceName, Namespace: resourceNamespace}}))).To(Succeed())
		})

		// Pods inherit labels from the Pod template, but the Deployment and Service
		// objects do not -- so `kubectl get deploy -l app=modelserver` returned
		// nothing until these were stamped explicitly.
		It("labels the Deployment and Service themselves, not just the Pods", func() {
			var deploy appsv1.Deployment
			Expect(k8sClient.Get(ctx, typeNamespacedName, &deploy)).To(Succeed())
			Expect(deploy.Labels).To(HaveKeyWithValue(appLabelKey, appLabelValue))
			Expect(deploy.Labels).To(HaveKeyWithValue(instanceLabelKey, resourceName))

			svc := getService()
			Expect(svc.Labels).To(HaveKeyWithValue(appLabelKey, appLabelValue))
			Expect(svc.Labels).To(HaveKeyWithValue(instanceLabelKey, resourceName))

			// The Service selector deliberately stays narrower than the labels.
			Expect(svc.Spec.Selector).To(Equal(map[string]string{
				instanceLabelKey: resourceName,
			}))
		})

		It("restores a Service selector that was edited away", func() {
			svc := getService()
			Expect(svc.Spec.Selector).To(HaveKeyWithValue(instanceLabelKey, resourceName))

			svc.Spec.Selector = map[string]string{instanceLabelKey: "someone-elses-pods"}
			Expect(k8sClient.Update(ctx, &svc)).To(Succeed())

			reconcileOnce()
			Expect(getService().Spec.Selector).
				To(HaveKeyWithValue(instanceLabelKey, resourceName))
		})

		// Drift WITHIN a list entry the controller owns. spec.ports is a
		// list-map keyed on (port, protocol), so retargeting port 8000 is a
		// genuine field-level conflict on an entry this controller declared,
		// and ForceOwnership resolves it in the controller's favour.
		//
		// Note the deliberate choice of targetPort over port. Editing the port
		// NUMBER changes the merge key, so SSA treats it as a different entry:
		// the controller re-adds its own {port: 8000} but cannot remove the
		// {port: 9999} another manager owns, and the Service ends up with both.
		// That is inherent to server-side apply, not something this controller
		// can fix -- see the note in reconcileService.
		It("restores a Service targetPort that was edited away", func() {
			svc := getService()
			Expect(svc.Spec.Ports).To(HaveLen(1))
			Expect(svc.Spec.Ports[0].TargetPort).To(Equal(intstr.FromInt(8000)))

			svc.Spec.Ports[0].TargetPort = intstr.FromInt(1234)
			Expect(k8sClient.Update(ctx, &svc)).To(Succeed())

			reconcileOnce()

			ports := getService().Spec.Ports
			Expect(ports).To(HaveLen(1))
			Expect(ports[0].Port).To(Equal(int32(8000)))
			Expect(ports[0].TargetPort).To(Equal(intstr.FromInt(8000)))
		})

		It("leaves the API-server-assigned clusterIP alone", func() {
			original := getService().Spec.ClusterIP
			Expect(original).NotTo(BeEmpty())

			// The controller never sets spec.clusterIP, so it owns no opinion about
			// it. A read-modify-write reconcile would have to copy this value across
			// or the update is rejected as immutable.
			reconcileOnce()
			Expect(getService().Spec.ClusterIP).To(Equal(original))
		})

		It("restores a Deployment image that was edited away", func() {
			var deploy appsv1.Deployment
			Expect(k8sClient.Get(ctx, typeNamespacedName, &deploy)).To(Succeed())
			original := deploy.Spec.Template.Spec.Containers[0].Image

			deploy.Spec.Template.Spec.Containers[0].Image = "nginx:tampered"
			Expect(k8sClient.Update(ctx, &deploy)).To(Succeed())

			reconcileOnce()

			Expect(k8sClient.Get(ctx, typeNamespacedName, &deploy)).To(Succeed())
			Expect(deploy.Spec.Template.Spec.Containers[0].Image).To(Equal(original))
		})

		It("passes the scheduler capacity through to the pod", func() {
			// This is the last link in the autoscaling chain:
			//   spec.maxBatchSize -> MAX_BATCH_SIZE -> Scheduler capacity
			//   -> a full batch queues -> num_requests_waiting > 0 -> HPA scales.
			// Without it the capacity is whatever the image defaults to, and any
			// threshold tuned against that metric is tuned against a constant.
			var ms servingv1alpha1.ModelServer
			Expect(k8sClient.Get(ctx, typeNamespacedName, &ms)).To(Succeed())
			Expect(ms.Spec.MaxBatchSize).To(BeNumerically(">", 0),
				"the CRD default must apply when the CR omits maxBatchSize")

			var deploy appsv1.Deployment
			Expect(k8sClient.Get(ctx, typeNamespacedName, &deploy)).To(Succeed())

			Expect(deploy.Spec.Template.Spec.Containers[0].Env).To(ContainElement(corev1.EnvVar{
				Name:  "MAX_BATCH_SIZE",
				Value: strconv.Itoa(int(ms.Spec.MaxBatchSize)),
			}))
		})

		It("re-applies MAX_BATCH_SIZE after it is edited away", func() {
			// Capacity is decided at process start, so changing it necessarily
			// restarts pods -- which makes drift here silently halve throughput
			// until the next unrelated rollout.
			var deploy appsv1.Deployment
			Expect(k8sClient.Get(ctx, typeNamespacedName, &deploy)).To(Succeed())
			original := deploy.Spec.Template.Spec.Containers[0].Env

			deploy.Spec.Template.Spec.Containers[0].Env = []corev1.EnvVar{
				{Name: "MAX_BATCH_SIZE", Value: "1"},
			}
			Expect(k8sClient.Update(ctx, &deploy)).To(Succeed())

			reconcileOnce()

			Expect(k8sClient.Get(ctx, typeNamespacedName, &deploy)).To(Succeed())
			Expect(deploy.Spec.Template.Spec.Containers[0].Env).To(ConsistOf(original))
		})
	})
})
