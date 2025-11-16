API Module
==========

The api module provides health check endpoints for monitoring and deployment.

Health Checks
-------------

.. automodule:: api.health
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Health Check
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from api.health import health_check_simple

   # Simple health check
   status = health_check_simple()
   print(status)  # {'status': 'healthy'}

Detailed Health Check
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from api.health import health_check_detailed

   # Detailed health check with GPU info
   info = health_check_detailed()
   print(f"Version: {info['version']}")
   print(f"Python: {info['python_version']}")
   print(f"GPU available: {info['gpu']['available']}")

   if info['gpu']['available']:
       for device in info['gpu']['devices']:
           print(f"GPU {device['id']}: {device['name']}")
           print(f"  Memory: {device['memory_free_gb']:.2f}GB free")

Kubernetes Probes
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from api.health import readiness_check, liveness_check

   # Readiness check
   readiness = readiness_check()
   if readiness['ready']:
       print("Service is ready to accept requests")
   else:
       print(f"Service not ready: {readiness['message']}")

   # Liveness check
   liveness = liveness_check()
   print(f"Service is alive: {liveness['status']}")

FastAPI Integration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fastapi import FastAPI
   from api.health import router

   # Create FastAPI app
   app = FastAPI(title="FluxPipeline API")

   # Include health check router
   app.include_router(router)

   # Now available at:
   # GET /health - simple health check
   # GET /health/detailed - detailed health info
   # GET /health/ready - readiness probe
   # GET /health/live - liveness probe

Docker Healthcheck
~~~~~~~~~~~~~~~~~~

.. code-block:: dockerfile

   # Add to your Dockerfile
   HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
       CMD python -c "from api.health import health_check_simple; health_check_simple()" || exit 1

Kubernetes Deployment
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Add to your Kubernetes deployment
   apiVersion: v1
   kind: Pod
   spec:
     containers:
     - name: fluxpipeline
       image: fluxpipeline:latest
       livenessProbe:
         httpGet:
           path: /health/live
           port: 7860
         initialDelaySeconds: 30
         periodSeconds: 10
       readinessProbe:
         httpGet:
           path: /health/ready
           port: 7860
         initialDelaySeconds: 30
         periodSeconds: 10
