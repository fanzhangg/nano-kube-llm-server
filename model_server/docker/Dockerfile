# Mock image: ~150MB, no torch. Still the CRD default for spec.image.
#
# Since Milestone 4 the mock is not a separate engine -- it is MockRunner driving
# the SAME Scheduler and the SAME run_loop as the GPU image. That is why
# batching.py imports no torch: the Kubernetes story (Pending/Loading/Ready,
# readiness gating, queue-depth metrics, autoscaling on num_requests_waiting)
# exercises the real admission rule here, not a simulation of it.
#
# Build from model_server/, not from docker/: the COPY path below is relative to
# the build context, and the sources live one level up from this file.
#
#   docker build -f docker/Dockerfile -t modelserver-mock:latest .
FROM python:3.12-slim
RUN pip install --no-cache-dir fastapi uvicorn
COPY main.py engines.py batching.py runners.py /app/
WORKDIR /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
