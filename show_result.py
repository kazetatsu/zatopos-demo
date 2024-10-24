import threading
import time

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import zatopos


def receive_task(n:int, ear_driver:zatopos.EarDriver, sounds:np.ndarray):
    # print("ear_driver on", n, "th t_receive:", ear_driver.c_driver)
    ear_driver.receive(sounds)
    return


def search_task(musical:zatopos.Musical, signal_spaces:np.ndarray, result:np.ndarray, l_result:threading.Lock):
    l_result.acquire(blocking=True)
    musical.search(signal_spaces, result)
    l_result.release()
    return


def calc_task(busno:int, devaddr:int, e_finish:threading.Event, result:np.ndarray, l_result:threading.Lock):
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
    signal_spaces = zatopos.get_signal_spaces(sounds_f)
    t_receive.join()
    sounds_f = sounds_u.transpose(0,2,1).astype(np.float32)

    # Main loop
    n = 3
    while True:
        t_receive = threading.Thread(target=receive_task, args=(n, ear_driver, sounds_u))
        t_search  = threading.Thread(target=search_task, args=(musical, signal_spaces, result, l_result))

        t_receive.start()
        t_search.start()

        signal_spaces = zatopos.get_signal_spaces(sounds_f)

        t_receive.join()
        sounds_f = sounds_u.transpose(0,2,1).astype(np.float32)
        n += 1
        t_search.join()

        if e_finish.is_set():
            break


    return


def main_task(busno:int, devaddr:int):
    e_finish = threading.Event()
    result = np.ndarray((8,8), dtype=np.float32)
    l_result = threading.Lock()

    t_calc = threading.Thread(target=calc_task, args=(busno, devaddr, e_finish, result, l_result))

    t_calc.start()

    fig = plt.figure()

    def update_func(frame, result:np.ndarray, l_result:threading.Lock):
        plt.cla()
        l_result.acquire()
        plt.imshow(result)
        l_result.release()

    fanim = anim.FuncAnimation(
        fig=fig,
        func=update_func,
        fargs=(result, l_result),
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
