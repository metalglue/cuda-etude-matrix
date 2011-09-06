
DEBUG = ARGUMENTS.get('debug', 'yes')
if DEBUG == 'yes' or DEBUG == 'true' or DEBUG == '1':
    DEBUG = True
else:
    DEBUG = False

env = Environment()
env.Tool('nvcc', toolpath = ['/home/kimura/usr/lib/scons/SCons/Tool/'])
env.Append(CPPPATH = ['/home/kimura/dist/cudasdk/C/common/inc'])
env.Append(LIBPATH  = ['/home/kimura/dist/cudasdk/C/lib', '/home/kimura/dist/cudasdk/C/common/lib/linux', '/usr/local/cuda/lib64'])
env.Append(LIBS = ['glut', 'GLEW_x86_64', 'cudart', 'cutil_x86_64'])
if DEBUG:
   env.Append(CCFLAGS='-g')
else:
   env.Append(CCFLAGS='-O')
env.Program(["ab.cu"])

