import pyopencl as cl

TASKS = 64

if __name__ == '__main__':

    print('create context')
    ctx = cl.create_some_context()

    print('create command queue')
    queue = cl.CommandQueue(ctx)

    print('load program from cl source file')
    f = open('hello_world_broken.cl', 'r', encoding='utf-8')
    kernels = ''.join(f.readlines())
    f.close()

    print('compile kernel code')
    prg = cl.Program(ctx, kernels).build()

    print('execute kernel programs')
    evt = prg.hello_world(queue, (TASKS, ), (1, ))
    print('wait for kernel executions')
    evt.wait()

    print('done')
