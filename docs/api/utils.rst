Utilities Module
================

The utils module provides utility functions for logging, system operations,
and performance metrics.

Logging Utilities
-----------------

.. automodule:: utils.logging_utils
   :members:
   :undoc-members:
   :show-inheritance:

System Utilities
----------------

.. automodule:: utils.system_utils
   :members:
   :undoc-members:
   :show-inheritance:

Performance Metrics
-------------------

.. automodule:: utils.metrics
   :members:
   :undoc-members:
   :show-inheritance:

PerformanceMetrics
~~~~~~~~~~~~~~~~~~

.. autoclass:: utils.metrics.PerformanceMetrics
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

Usage Examples
--------------

Performance Logging
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from utils import log_performance

   # Automatic performance logging
   @log_performance
   def generate_images(prompt):
       # Your generation code
       pass

System Utilities
~~~~~~~~~~~~~~~~

.. code-block:: python

   from utils import setup_workspace, suppress_warnings

   # Setup workspace
   workspace = setup_workspace()
   print(f"Workspace: {workspace}")

   # Suppress warnings
   suppress_warnings()

Performance Metrics
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from utils import PerformanceMetrics

   # Initialize metrics
   metrics = PerformanceMetrics()

   # Use context manager
   with metrics.measure("image_generation", "generation"):
       image = generate_image(prompt)

   # Get statistics
   stats = metrics.get_operation_stats("image_generation")
   print(f"Average time: {stats['avg_time_seconds']:.2f}s")
   print(f"Count: {stats['count']}")

   # Export metrics
   metrics.export_to_json("metrics.json")
   metrics.export_to_csv("metrics.csv")

   # Generate report
   report = metrics.report()
   print(report)

Manual Metric Tracking
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from utils import PerformanceMetrics

   metrics = PerformanceMetrics()

   # Start tracking
   metrics.start_operation("model_load")
   load_model()
   result = metrics.end_operation("model_load", "model_load")

   print(f"Model load time: {result['duration_seconds']:.2f}s")
   print(f"Memory used: {result['cpu_memory_delta_gb']:.2f}GB")
