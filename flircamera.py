import time, platform, json
import os.path as path
from multiprocessing import shared_memory
import numpy as np
import PySpin
try:
    import ffmpeg
except importError:
    print('If you want to use ffmpeg backend to save video, $ pip install ffmpeg-python')


class NodeMap(object):
    '''
    Helper class wrapping around nodemap for easier access.
    '''
    casters = dict(
        integer = PySpin.CIntegerPtr,
        float = PySpin.CFloatPtr,
        boolean = PySpin.CBooleanPtr,
        command = PySpin.CCommandPtr,
        enumeration = PySpin.CEnumerationPtr,
        category = PySpin.CCategoryPtr,
        value = PySpin.CValuePtr,
    )
    
    kinds = dict(
        AcquisitionFrameRate = 'float',
        AcquisitionFrameRateEnable = 'boolean',
        AcquisitionMode = 'enumeration',
        BinningHorizontal = 'integer',
        BinningVertical = 'integer',
        DeviceInformation = 'category',
        ExposureAuto = 'enumeration',
        ExposureTime = 'float',
        Gain = 'float',
        GainAuto = 'enumeration',
        PixelFormat = 'enumeration',
    )
    
    def __init__(self, nodemap):
        self.nodemap = nodemap
        
    def get(self, name, kind=None):
        if kind is None:
            kind = self.kinds.get(name)
        node = self.nodemap.GetNode(name)
        node = self.casters[kind](node)
        if PySpin.IsAvailable(node) and PySpin.IsReadable(node):
            if kind in ['integer', 'float', 'boolean']:
                value = node.GetValue()
            elif kind == 'command':
                value = node.GetToolTip()
            elif kind == 'enumeration':
                value = node.ToString()
            elif kind == 'category':
                value = {}
                for feature in node.GetFeatures():
                    feature_node = PySpin.CValuePtr(feature)
                    value[feature_node.GetName()] = feature_node.ToString()
        else:
            raise ValueError(f'"{node.GetDisplayName()}" is not available.')
        return value
    
    def get_node(self, name, kind=None):
        if kind is None:
            kind = self.kinds.get(name)
        node = self.nodemap.GetNode(name)
        node = self.casters[kind](node)
        if PySpin.IsAvailable(node) and PySpin.IsReadable(node):
            return node
        else:
            raise ValueError(f'"{node.GetDisplayName()}" is not available.')

    def set(self, name, value, kind=None):
        if kind is None:
            kind = self.kinds.get(name)
        node = self.nodemap.GetNode(name)
        node = self.casters[kind](node)
        if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
            if kind in ['integer', 'float', 'boolean']:
                node.SetValue(value)
            elif kind == 'enumeration':
                entry_node = node.GetEntryByName(value)
                if PySpin.IsAvailable(entry_node) and PySpin.IsReadable(entry_node):
                    node.SetIntValue(entry_node.GetValue())
                else:
                    raise ValueError(f'Unable to set "{node.GetDisplayName()}" to "{value}" (@{entry_node})')
        else:
            raise ValueError(f'Unable to set "{node.GetDisplayName()}" to "{value}" (@{node})')


class SingleCamera(object):
    '''
    Select, config and use a single camera in the system.
    '''
    def __init__(self, idx=0, config=None, verbose=1):
        self.system = PySpin.System.GetInstance()
        self.verbose = verbose
        if self.verbose > 0:
            version = self.system.GetLibraryVersion()
            print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))
        self.camera = None
        self.nodemap = None
        if idx is not None:
            self.select_camera(idx)
            self.config_camera(config)

    def select_camera(self, idx=0):
        cam_list = self.system.GetCameras()
        if self.verbose > 0 or idx is None:
            print(f"Detect {len(cam_list)} camera{'s' if len(cam_list)>1 else ''}")
        if len(cam_list) > 0:
            if idx is None: # Print available cameras
                print('-'*20)
                for k, cam in enumerate(cam_list):
                    nodemap_TL = NodeMap(cam.GetTLDeviceNodeMap())
                    device_info = nodemap_TL.get('DeviceInformation')
                    print(f"#{k}: {device_info['DeviceDisplayName']} ({device_info['DeviceSerialNumber']})")
            else: # Select specified camera
                self.camera = cam_list[idx]
                nodemap_TL = NodeMap(self.camera.GetTLDeviceNodeMap())
                device_info = nodemap_TL.get('DeviceInformation')
                if self.verbose > 0:
                    print(f"Select camera #{idx}: {device_info['DeviceDisplayName']} ({device_info['DeviceSerialNumber']})")
                # Init select camera
                self.camera.Init() # Otherwise nodemap will be NULL
                self.nodemap = NodeMap(self.camera.GetNodeMap())
        cam_list.Clear()

    def __getitem__(self, key):
        if self.nodemap is not None:
            return self.nodemap.get(key)
        else:
            raise ValueError('Camera nodemap not populated yet. You need to call select_camera() first.')

    def __setitem__(self, key, value):
        if self.nodemap is not None:
            self.nodemap.set(key, value)
        else:
            raise ValueError('Camera nodemap not populated yet. You need to call select_camera() first.')

    def config_camera(self, config=None):
        '''
        Notes
        -----
        1. These settings must be applied before BeginAcquisition(); otherwise, they will be read only. 
        2. Settings are applied immediately. So order matters.
        '''
        if config is None:
            # Minimal setting: continuous acquisition frame rate
            self['AcquisitionMode'] = 'Continuous'
            self['AcquisitionFrameRateEnable'] = True
            self['AcquisitionFrameRate'] = self.nodemap.get_node('AcquisitionFrameRate').GetMax()
        else:
            if isinstance(config, str): # JSON file name
                with open(config, 'r') as json_file:
                    config = json.load(json_file)
            for key, value in config.items():
                self[key] = value

    def begin_acquisition(self, fname, record_kws=None):
        '''
        Parameters
        ----------
        fname : str
        record_kws : dict
            dict(backend=None, codec='h264', bitrate=None, quality=None)
        '''
        self.camera.Init()
        self.camera.BeginAcquisition()
        # Get frame dimension
        image = self.camera.GetNextImage(1000) # block until grabTimeout=1000 ms
        image.Release()
        width = image.GetWidth()
        height = image.GetHeight()
        # Get frame rate
        self._frame_rate = self['AcquisitionFrameRate']
        # Get auto exposure time and gain
        if self['ExposureAuto'] in ['Once', 'Continuous']:
            print(f"Auto ({self['ExposureAuto']}) ExposureTime = {self['ExposureTime']:g}")
        if self['GainAuto'] in ['Once', 'Continuous']:
            print(f"Auto ({self['GainAuto']}) Gain = {self['Gain']:g}")
        # Create shared memory
        self.shared_buffer = SharedBuffer([10, height, width])
        # Create VideoRecorder()
        if record_kws is None:
            record_kws = dict()
        self.recorder = VideoRecorder(fname, width, height, self._frame_rate, **record_kws)
        self._frame_count = 0
        self._start_time = time.time()

    def get_frame(self):
        image = self.camera.GetNextImage(1000) # block until grabTimeout=1000 ms
        if image.IsIncomplete():
            raise RuntimeError(f"Image incomplete with image status {image.GetImageStatus()}")
        else:
            im = image.GetNDArray()
            self.shared_buffer.latest = im
            self.recorder.add_frame(image)
            image.Release()
        self._frame_count += 1
        return im

    def end_acquisition(self):
        duration = time.time() - self._start_time
        self.camera.EndAcquisition()
        self.recorder.finalize()
        self.shared_buffer.finalize()
        if self.verbose:
            print(f"Acquired {self._frame_count} images")
            print(f"All data saved to '{path.realpath(self.recorder.fname)}'")
            print(f"Target frame rate = {self._frame_rate:g} Hz, measured frame rate = {self._frame_count/duration:g} Hz")

    def finalize(self):
        if self.camera is not None:
            self.camera.DeInit()
            del self.camera
        self.system.ReleaseInstance()
 

class VideoRecorder(object):
    def __init__(self, fname, width, height, frame_rate, backend=None, codec='h264', bitrate=None, quality=None):
        '''
        Compress and write acquired image frames to the hard disk on-the-fly.

        Parameters
        ----------
        fname : str
            Output video file name.
        backend : str
            'spinnaker': Native API provided by the camera vendor.
                Very efficient, but not available on macOS.
            'ffmpeg': Need to install ffmpeg and ffmpeg-python.
                Not fully optimized, but available across all major platform.
            'numpy': Save uncompressed data in memory as a list of numpy array.
                No compression. Quickly eat up huge amount of memory.
        codec : str
            'rawvideo' : uncompressed
            'mjpeg': Motion JPEG (intraframe-only compression scheme, <1:20 compression)
            'h264': H.264 (modern interframe video codec, >1:50 compression)
        '''
        self.fname = fname
        self.width = width
        self.height = height
        self.frame_rate = frame_rate
        if backend is None:
            backend = 'ffmpeg' if platform.system() == 'Darwin' else 'spinnaker'
        self.backend = backend
        if self.backend == 'spinnaker':
            self.recorder = PySpin.SpinVideo()
            if codec == 'rawvideo':
                option = PySpin.AVIOption()
            elif codec == 'mjpeg':
                option = PySpin.MJPGOption()
                if quality is None:
                    quality = 75
                option.quality = quality
            elif codec == 'h264':
                option = PySpin.H264Option()
                option.width = self.width
                option.height = self.height
                if bitrate is None:
                    bitrate = 1000000
                if bitrate < 1: # Compression rate, e.g., 1:50
                    bitrate = int(self.raw_bitrate * bitrate)
                option.bitrate = bitrate
            option.frameRate = self.frame_rate
            if path.splitext(self.fname)[1] in ['.avi', '.AVI']:
                self.fname = self.fname[:-4] # spinnaker will always append .avi
            self.recorder.Open(self.fname, option)
        elif self.backend == 'ffmpeg':
            stream = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='gray', 
                s=f"{self.width}x{self.height}", framerate=self.frame_rate)
            if bitrate is not None:
                if bitrate < 1: # Compression rate, e.g., 1:50
                    bitrate = int(self.raw_bitrate * bitrate)
                stream = stream.output(self.fname, pix_fmt='yuv420p', vcodec=codec, video_bitrate=bitrate) 
            elif quality is not None:
                stream = stream.output(self.fname, pix_fmt='yuv420p', vcodec=codec, crf=quality)
            else:
                stream = stream.output(self.fname, pix_fmt='yuv420p', vcodec=codec) # crf=23
            self.recorder = stream.overwrite_output().run_async(pipe_stdin=True)
        elif self.backend == 'numpy':
            self.recorder = []

    raw_bitrate = property(lambda self: self.width*self.height*self.frame_rate*8)

    def add_frame(self, image):
        if self.backend == 'spinnaker':
            im = image.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
            self.recorder.Append(im)
        elif self.backend == 'ffmpeg':
            im = image.GetNDArray()
            self.recorder.stdin.write(im.tobytes())
        elif self.backend == 'numpy':
            im = image.GetNDArray()
            self.recorder.append(im)
        
    def finalize(self):
        if self.backend == 'spinnaker':
            self.recorder.Close()
        elif self.backend == 'ffmpeg':
            self.recorder.stdin.close()
            self.recorder.wait()
        elif self.backend == 'numpy':
            return self.recorder


class SharedBuffer(object):
    def __init__(self, shape=None, prefix=None):
        n_states = 4 # length, height, width, index
        if prefix is None:
            prefix = 'flircamera_sharedbuffer'
        if shape is not None:
            self.master = True
            self._shm_states = shared_memory.SharedMemory(name=f'{prefix}_states', 
                create=True, size=n_states * np.dtype('int').itemsize)
            self.states = np.ndarray(n_states, dtype=int, buffer=self._shm_states.buf)
            self.states[:3] = shape
            self.states[3] = 0
            self._shm_buffer = shared_memory.SharedMemory(name=f'{prefix}_buffer', 
                create=True, size=np.prod(shape) * np.dtype('uint8').itemsize)
            self.buffer = np.ndarray(shape, dtype=np.uint8, buffer=self._shm_buffer.buf)
        else:
            self.master = False
            self._shm_states = shared_memory.SharedMemory(name=f'{prefix}_states', 
                create=False, size=n_states * np.dtype('int').itemsize)
            self.states = np.ndarray(n_states, dtype=int, buffer=self._shm_states.buf)
            shape = self.states[:3]
            self._shm_buffer = shared_memory.SharedMemory(name=f'{prefix}_buffer', 
                create=False, size=np.prod(shape) * np.dtype('uint8').itemsize)
            self.buffer = np.ndarray(shape, dtype=np.uint8, buffer=self._shm_buffer.buf)

    def finalize(self):
        # TODO: How to do this properly??
        del self.states
        del self.buffer
        self._shm_states.close()
        self._shm_buffer.close()
        if self.master:
            self._shm_states.unlink()
            self._shm_buffer.unlink()

    index = property(lambda self: self.states[3])
    @index.setter
    def index(self, value):
        self.states[3] = value

    latest = property(lambda self: self.buffer[self.index])
    @latest.setter
    def latest(self, value):
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.states[0]


if __name__ == '__main__':
    # The following is a quick demonstration of the main usage
    import argparse, textwrap
    parser = argparse.ArgumentParser(description='Acquire images, save as video, and preview online.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
        Examples
        --------
        In the first terminal, start image acquisition:
        $ python flircamera.py -o test.avi -c highspeed.json -n 10000

        And then in the second terminal, start online preview:
        $ python flircamera.py -v
        '''))
    parser.add_argument('-o', '--output', default='test.avi', help='output video filename')
    parser.add_argument('-c', '--config', default='config.json', help='camera config filename')
    parser.add_argument('-v', '--viewer', action='store_true', help='open a preview window')
    parser.add_argument('-n', '--n_frames', type=int, default=1000, help='number of frames to acquire')
    args = parser.parse_args()

    if args.viewer: # Online preview
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.use('Qt5Agg')
        fig = plt.figure()
        fig.has_been_closed = False
        def on_close(event):
            event.canvas.figure.has_been_closed = True
        fig.canvas.mpl_connect('close_event', on_close)

        shared_buffer = SharedBuffer()
        img = plt.imshow(shared_buffer.latest/255.0, vmin=0, vmax=1, cmap='gray')
        plt.show(block=False)
        while not fig.has_been_closed:
            img.set_data(shared_buffer.latest/255.0)
            fig.canvas.draw()
            fig.canvas.flush_events()
        shared_buffer.finalize()
    else: # Acquire images
        camera = SingleCamera(config=args.config)
        camera.begin_acquisition(args.output, record_kws=dict(bitrate=1e6))
        for k in range(args.n_frames):
            im = camera.get_frame()
        camera.end_acquisition()
        camera.finalize()

