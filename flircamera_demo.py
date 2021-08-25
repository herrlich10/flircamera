import flircamera

camera = flircamera.SingleCamera(config='highspeed.json', share=False)
camera.begin_acquisition('test.avi', record_kws=dict(bitrate=1e6))
for k in range(200*60*5):
    im = camera.get_frame()
camera.end_acquisition()
camera.finalize()