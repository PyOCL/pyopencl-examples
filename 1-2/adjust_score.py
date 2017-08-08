import math
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
    # prepare memory for final answer from OpenCL
    final = numpy.zeros(TASKS, dtype=numpy.int32)
    dev_fianl = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, final.nbytes)
    # prepare data for comparison.
    correct = numpy.zeros(TASKS, dtype=numpy.int32)
    for i in range(0, TASKS):
        correct[i] = math.floor(math.sqrt(matrix[i]) * 10)

    print('create command queue')
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    print('load program from cl source file')
    f = open('adjust_score.cl', 'r')
    kernels = ''.join(f.readlines())
    f.close()

    print('compile kernel code')
    prg = cl.Program(ctx, kernels).build();

    print('execute kernel programs')
    evt = prg.adjust_score(queue, (TASKS, ), (1, ), dev_matrix, dev_fianl)
    print('wait for kernel executions')
    evt.wait();
    elapsed = 1e-9 * (evt.profile.end - evt.profile.start)

    print('elapsed time: {}'.format(elapsed))

    cl.enqueue_read_buffer(queue, dev_fianl, final).wait()
    equal = numpy.all(correct == final)
    print(final)
    if not equal:
        print('Results doesnot match!!')
    else:
        print('Results is OK')

