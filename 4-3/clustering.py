#!/usr/bin/python3
import os
import time
import random
import numpy
import pyopencl as cl
import pyopencl.array

def plot_grouping_result(point_cids, group_ids, point_info):
    assert len(point_cids) != 0
    import matplotlib.pyplot as plt
    markers = ['p', '*', '+', 'x', 'd', 'o', 'v', 's', 'h']
    colors = [(random.random(), random.random(), random.random()) for x in range(len(point_cids))]
    while len(point_cids) > 0:
        c_id = point_cids.pop()
        clr = colors.pop()
        makr = markers[random.randint(0, len(markers)-1)]
        x = []
        y = []
        for idx, gid in enumerate(group_ids):
            if gid == c_id:
                x.append(point_info[idx][0])
                y.append(point_info[idx][1])
        plt.scatter(x, y, color=clr, marker=makr)

    plt.ylabel('y')
    plt.xlabel('x')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    print('load program from cl source file')
    f = open('clustering.cl', 'r', encoding='utf-8')
    kernels = ''.join(f.readlines())
    f.close()

    print('prepare data ... ')
    # The number of points randomly generated
    random.seed()
    print(">>> How many points to be clustered ? ")
    strNum = input()
    num_points = int(strNum) if strNum != '' else 0
    assert num_points >= 1, "Please input a number >= 1"

    point_ids = list(range(0, num_points))
    point_info = {point_id: (random.random() * 100, random.random() * 100) for point_id in point_ids}
    pointX = [point_info[v][0] for v in point_info]
    pointY = [point_info[v][1] for v in point_info]

    # The number of group you want to divide.
    print(">>> How many clusters you want ? ")
    strGroup = input()
    num_of_groups = int(strGroup) if strGroup != '' else 0
    assert num_of_groups >= 1, "Please input a number >= 1."
    assert num_points >= num_of_groups, "Number of points should >= number of clusters."

    group_id_set = list(range(0, num_of_groups))
    cluster_centers_X = []
    cluster_centers_Y = []
    for idx in group_id_set:
        cluster_centers_X.append(pointX[idx])
        cluster_centers_Y.append(pointY[idx])
    cluster_ids = []
    for x in range(num_points):
        cluster_ids.append(x if x < num_of_groups else -1)

    start_time = time.time()
    # prepare host memory for OpenCL
    np_centers_x = numpy.array(cluster_centers_X, dtype=numpy.float32)
    np_centers_y = numpy.array(cluster_centers_Y, dtype=numpy.float32)
    np_point_x = numpy.array(pointX, dtype=numpy.float32)
    np_point_y = numpy.array(pointY, dtype=numpy.float32)
    np_clusters_ids = numpy.array(cluster_ids, dtype=numpy.int32)
    time_hostdata_loaded = time.time()

    # create opencl context & queue
    print('create context ...')
    ctx = cl.create_some_context()
    print('create command queue ...')
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    time_ctx_queue_creation = time.time()

    # prepare device memory for OpenCL
    print('prepare device memory for input / output')
    dev_centers_x = cl.array.to_device(queue, np_centers_x)
    dev_centers_y = cl.array.to_device(queue, np_centers_y)
    dev_points_x = cl.array.to_device(queue, np_point_x)
    dev_points_y = cl.array.to_device(queue, np_point_y)
    dev_clusters_id = cl.array.to_device(queue, np_clusters_ids)
    time_devicedata_loaded = time.time()

    print('compile kernel code')
    prg = cl.Program(ctx, kernels).build()
    time_kernel_compilation = time.time()

    np_num_of_clusters = numpy.int32(num_of_groups)
    np_num_of_points = numpy.int32(num_points)
    print('execute kernel programs')
    print('wait for kernel executions')
    elapsed = 0
    last_cluster_ids = None
    time_data_readback_total = 0
    for i in range(10000):
        evt = prg.do_clustering(queue, (num_points,), None,
                                np_num_of_clusters, np_num_of_points,
                                dev_centers_x.data, dev_centers_y.data,
                                dev_points_x.data, dev_points_y.data,
                                dev_clusters_id.data)
        evt.wait()
        elapsed += (1e-9 * (evt.profile.end - evt.profile.start))
        evt = prg.calc_centroid(queue, (num_of_groups,), None,
                                np_num_of_clusters, np_num_of_points,
                                dev_centers_x.data, dev_centers_y.data,
                                dev_points_x.data, dev_points_y.data,
                                dev_clusters_id.data)
        evt.wait()
        elapsed += (1e-9 * (evt.profile.end - evt.profile.start))

        time_before_readback = time.time()
        tmp_clusters_ids = dev_clusters_id.get()
        time_data_readback_total += (time.time() - time_before_readback)
        if numpy.array_equal(tmp_clusters_ids, last_cluster_ids):
            print("break ........... @ {}".format(i))
            break
        else:
            last_cluster_ids = tmp_clusters_ids

    time_before_readback = time.time()
    cids = dev_clusters_id.get()
    time_data_readback_total = (time.time() - time_before_readback)

    print('Prepare host data took       : {}'.format(time_hostdata_loaded - start_time))
    print('Create CTX/QUEUE took        : {}'.format(time_ctx_queue_creation - time_hostdata_loaded))
    print('Upload data to device took   : {}'.format(time_devicedata_loaded - time_ctx_queue_creation))
    print('Compile kernel took          : {}'.format(time_kernel_compilation - time_devicedata_loaded))
    print('OpenCL elapsed time          : {}'.format(elapsed))
    print('Offload data from device took: {}'.format(time_data_readback_total))

    plot_grouping_result(point_ids, cids, point_info)
    print('Results is OK')
