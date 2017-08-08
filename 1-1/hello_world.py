import numpy
import pyopencl as cl

TASKS = 64

if __name__ == '__main__':

    print('create context')
    ctx = cl.create_some_context()

    print('prepare data')
    matrix = numpy.random.randint(low=1, high=101, dtype=numpy.int32, size=TASKS)
    dev_matrix = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix)
    print(matrix)

    print('create command queue')
    queue = cl.CommandQueue(ctx)

    print('load program from cl source file')
    f = open('hello_world.cl', 'r')
    kernels = ''.join(f.readlines())
    f.close()

    print('compile kernel code')
    prg = cl.Program(ctx, kernels).build();

    print('execute kernel programs')
    evt = prg.hello_world(queue, (TASKS, ), (1, ), dev_matrix)
    print('wait for kernel executions')
    evt.wait();

    print('done')
