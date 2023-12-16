import ffmpeg
import IPython
import base64


def pipe_to_ffmpeg(filename):
  args = (
        ffmpeg
        .input('pipe:')
        .output(filename)
        .overwrite_output()
        .compile()
    )
  return subprocess.Popen(args, stdin=subprocess.PIPE)

def embed_mp4(filename):
  """Embeds an mp4 file in the notebook."""
  video = open(filename,'rb').read()
  b64 = base64.b64encode(video)
  tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

  return IPython.display.HTML(tag)