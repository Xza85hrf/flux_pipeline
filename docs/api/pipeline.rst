Pipeline Module
===============

The pipeline module contains the main FluxPipeline class for image generation.

FluxPipeline
------------

.. automodule:: pipeline.flux_pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Main Pipeline Class
~~~~~~~~~~~~~~~~~~~

.. autoclass:: pipeline.flux_pipeline.FluxPipeline
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   from pipeline import FluxPipeline
   from utils import setup_workspace

   # Initialize
   workspace = setup_workspace()
   pipeline = FluxPipeline(workspace=workspace)

   # Load model
   if pipeline.load_model():
       # Generate image
       image, seed = pipeline.generate_image(
           prompt="A beautiful landscape",
           num_inference_steps=4,
           height=1024,
           width=1024
       )

       if image:
           image.save(f"output_{seed}.png")
