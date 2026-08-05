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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// ModelServerSpec defines the desired state of ModelServer
type ModelServerSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	// The following markers will use OpenAPI v3 schema to validate the value
	// More info: https://book.kubebuilder.io/reference/markers/crd-validation.html

	// model is the HuggingFace repo id to serve, e.g. "Qwen/Qwen3-0.6B". It is
	// passed to the pod as both MODEL_ID (which selects the real engine) and
	// MODEL_NAME (the label reported in /metrics and completion responses).
	// +kubebuilder:validation:MinLength=1
	Model string `json:"model"`

	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:default=1
	Replicas int32 `json:"replicas,omitempty"`

	// +kubebuilder:default="modelserver-mock:latest"
	Image string `json:"image,omitempty"`

	// gpus requests whole NVIDIA GPUs for each replica. Zero (the default) runs
	// on CPU, which is slower but needs no device plugin -- every part of the
	// control loop this operator implements works identically either way.
	//
	// Set only as a limit: nvidia.com/gpu is a non-overcommittable extended
	// resource, so Kubernetes requires requests and limits to match and rejects
	// specifying them separately.
	// +kubebuilder:validation:Minimum=0
	// +optional
	GPUs int32 `json:"gpus,omitempty"`

	// maxBatchSize is how many sequences the scheduler may run concurrently. This is
	// the real KV-cache capacity limit, and it is what decides whether an arriving
	// request starts or queues -- which is what makes vllm:num_requests_waiting a
	// measurement rather than a number the server invented.
	//
	// Set it too high and you OOM the GPU; too low and you leave throughput on the
	// table. It is the one knob this CRD exposes that trades memory for throughput.
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=8
	// +optional
	MaxBatchSize int32 `json:"maxBatchSize,omitempty"`
}

// ModelServerStatus defines the observed state of ModelServer.
type ModelServerStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// For Kubernetes API conventions, see:
	// https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md#typical-status-properties

	// conditions represent the current state of the ModelServer resource.
	// Each condition has a unique type and reflects the status of a specific aspect of the resource.
	//
	// Standard condition types include:
	// - "Available": the resource is fully functional
	// - "Progressing": the resource is being created or updated
	// - "Degraded": the resource failed to reach or maintain its desired state
	//
	// The status of each condition is one of True, False, or Unknown.
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// +optional
	ReadyReplicas int32 `json:"readyReplicas,omitempty"`

	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// +kubebuilder:validation:Enum=Pending;Loading;Ready
	// +optional
	Phase string `json:"phase,omitempty"`
}

// ModelServer is the Schema for the modelservers API
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Ready",type=integer,JSONPath=`.status.readyReplicas`
type ModelServer struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is a standard object metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitzero"`

	// spec defines the desired state of ModelServer
	// +required
	Spec ModelServerSpec `json:"spec"`

	// status defines the observed state of ModelServer
	// +optional
	Status ModelServerStatus `json:"status,omitzero"`
}

// +kubebuilder:object:root=true

// ModelServerList contains a list of ModelServer
type ModelServerList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitzero"`
	Items           []ModelServer `json:"items"`
}

const (
	ConditionTypeAvailable   = "Available"
	ConditionTypeProgressing = "Progressing"
)

func init() {
	SchemeBuilder.Register(func(s *runtime.Scheme) error {
		s.AddKnownTypes(SchemeGroupVersion, &ModelServer{}, &ModelServerList{})
		return nil
	})
}
