import threading
import time

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import zatopos


def receive_task(ear_driver:zatopos.EarDriver, sounds:np.ndarray):
    ear_driver.receive(sounds)
    return


def search_task(musical:zatopos.Musical, signal_spaces:np.ndarray, result:np.ndarray, l_result:threading.Lock):
    l_result.acquire(blocking=True)
    musical.search(signal_spaces, result)
    l_result.release()
    return


def calc_task(
    busno:int, devaddr:int,
    e_finish:threading.Event,
    snratio:np.ndarray, l_snratio:threading.Lock, result:np.ndarray, l_result:threading.Lock
):
    # Reusing variables
    ear_driver = zatopos.EarDriver(busno, devaddr)
    # print("ear_driver on t_calc: ", ear_driver.c_driver)
    musical = zatopos.Musical()
    sounds_u = np.ndarray((5, zatopos.EAR_WINDOW_LEN, zatopos.EAR_NUM_MICS), dtype=np.uint16) # BUG: shape[0] を奇数にするとタイムアウト起こす

    # Init 1
    receive_task(1, ear_driver, sounds_u)
    sounds_f = sounds_u.transpose(0,2,1).astype(np.float32)

    # Init 2
    t_receive = threading.Thread(target=receive_task, args=(2, ear_driver, sounds_u))
    t_receive.start()

    eigval, signal_spaces = zatopos.fft_eig(sounds_f)
    signal_spaces /= np.sqrt(signal_spaces * signal_spaces.conjugate(), dtype=np.complex64)
    l_snratio.acquire()
    snratio[:] = eigval[:,0] / eigval.sum(axis=1)
    l_snratio.release()

    t_receive.join()
    sounds_f = sounds_u.transpose(0,2,1).astype(np.float32)

    # Main loop
    while True:
        t_receive = threading.Thread(target=receive_task, args=(ear_driver, sounds_u))
        t_search  = threading.Thread(target=search_task, args=(musical, signal_spaces, result, l_result))

        t_receive.start()
        t_search.start()

        eigval, signal_spaces = zatopos.fft_eig(sounds_f)
        signal_spaces /= np.sqrt(signal_spaces * signal_spaces.conjugate(), dtype=np.complex64)
        l_snratio.acquire()
        snratio[:] = eigval[:,0] / eigval.sum(axis=1)
        l_snratio.release()

        t_receive.join()
        sounds_f = sounds_u.transpose(0,2,1).astype(np.float32)
        t_search.join()

        if e_finish.is_set():
            break


    return


def main_task(busno:int, devaddr:int):
    e_finish = threading.Event()
    n = int(zatopos.EAR_WINDOW_LEN/2-1)
    snratio = np.ndarray((n,), dtype=np.float32)
    l_snratio = threading.Lock()
    result = np.ndarray((n,8,8), dtype=np.float32)
    l_result = threading.Lock()

    t_calc = threading.Thread(
        target=calc_task,
        args=(
            busno, devaddr,
            e_finish,
            snratio, l_snratio, result, l_result
        )
    )

    t_calc.start()

    fig = plt.figure()
    ax_sn = fig.add_subplot(2,1,1)
    ax_res = fig.add_subplot(2,1,2)

    def update_func(
        frame,
        ax_sn:plt.Axes, snratio:np.ndarray, l_snratio:threading.Lock,
        ax_res:plt.Axes, result:np.ndarray, l_result:threading.Lock
    ):
        # Show S/N ratio
        n = int(zatopos.EAR_WINDOW_LEN/2-1)
        x = np.arange(n)
        ax_sn.cla()
        ax_sn.set_xlim(xmin=0, xmax=n)
        ax_sn.set_ylim(ymin=0.0, ymax=1.0)
        l_snratio.acquire()
        ax_sn.plot(x, snratio)
        i_max_sn = np.argmax(snratio)
        max_sn = snratio[i_max_sn]
        l_snratio.release()
        ax_sn.scatter(i_max_sn, max_sn)

        # Show result
        ax_res.cla()
        l_result.acquire()
        ax_res.imshow(result[i_max_sn])
        l_result.release()

    fanim = anim.FuncAnimation(
        fig=fig,
        func=update_func,
        fargs=(ax_sn, snratio, l_snratio, ax_res, result, l_result),
        interval=200,
        frames=range(32),
        repeat=True
    )

    plt.show()

    e_finish.set()

    t_calc.join()

    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("busno", type=int)
    parser.add_argument("devaddr", type=int)
    args = parser.parse_args()

    main_task(args.busno, args.devaddr)
