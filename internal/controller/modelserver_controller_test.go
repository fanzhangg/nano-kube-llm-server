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

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
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
})
