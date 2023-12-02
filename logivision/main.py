import threading
from moving import keyboard_listener, scrolling_code
from demo import demo
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_id", type=int, default=0)
    args = parser.parse_args()
    
    demo_thread = threading.Thread(target=demo(args))
    keyboard_thread = threading.Thread(target = keyboard_listener)
    scrolling_thread = threading.Thread(target=scrolling_code)
    
    demo_thread.start()
    scrolling_thread.start()
    keyboard_thread.start()
    

if __name__ == "__main__":
    main()
