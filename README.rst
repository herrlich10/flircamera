Introduction
============

This package is a convenient wrapper around the Spinnaker SDK for FLIR cameras.

It allows you to query available cameras connected to your system, initialize 
(read and write) camera settings, acquire images in a for-loop of your script 
while compress the frames into a compact video (e.g., H.264) file on-the-go and
allow for online image preview from another process via memory sharing mechanism.

Installation 
============

Just copy the ``flircamera.py`` file (and perhaps necessary config json file) 
to your project folder.

Dependencies
============

- Spinnaker SDK and its Python binding ``spinnaker-python``
- ``ffmpeg`` and its Python binding ``ffmpeg-python`` (optional, only required if you need to record video on macOS) 

Quick starts
============

Basic usage is as follows:

.. code-block:: python

    import flircamera

    camera = flircamera.SingleCamera(config='config.json', share=False)
    camera.begin_acquisition('test.avi', record_kws=dict(bitrate=1e6))
    for k in range(1000):
        im = camera.get_frame()
    camera.end_acquisition()
    camera.finalize()

For a demonstration of online preview via shared memory, run

.. code-block:: shell

    $ python flircamera.py -h

and read the examples.