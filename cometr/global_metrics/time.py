import time

def func():
    start = time.time() # create a time for the start of the process

    print('doing stuff')
    time.sleep(5)

    end = time.time() # create and object for the end of the process

    runtime = end - start

    print(f'total runtime in seconds: {runtime}')

    return

if __name__ == '__main__':
    func()