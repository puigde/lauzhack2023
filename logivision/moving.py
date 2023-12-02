import pyautogui
import pyperclip
from pynput import keyboard
import time


#current = set()

screen_width, screen_height = pyautogui.size()
edge_distance = 50
scroll = False
#event = threading.Event()

def obtener_posicion_raton():
    return pyautogui.position()

def obtener_parrafo_en_posicion(posicion):
    #pyautogui.click(posicion)
    pyautogui.click(posicion, clicks=3, interval=0.2)  # 3 clics con intervalo de 0.2 segundos
    time.sleep(0.2)
    pyautogui.hotkey('ctrl', 'c') # Copia al portapapeles
    return pyperclip.paste()  # Obtiene el texto del portapapeles

def is_at_end():
    x,y = obtener_posicion_raton()
    return y > screen_height - edge_distance

def keyboard_listener():
    global scroll
    def on_press(key):
        global scroll
        try:
            
            #if 'c' in current and keyboard.Key.ctrl in current and done:
            if key == keyboard.Key.f2:
                posicion_ratón = obtener_posicion_raton()
                párrafo = obtener_parrafo_en_posicion(posicion_ratón)
                print(f"Párrafo copiado en la posición {posicion_ratón}:\n{párrafo}")
            
            if key == keyboard.Key.f4:
                scroll = not scroll
                #event.set()
                
        except AttributeError:
            pass

    def on_release(key):
        pass
    print("Presiona F2 para copiar el párrafo en la posición actual del ratón.")
        
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def scrolling_code():
    global scroll
    while True:
        #event.wait()
        if scroll and is_at_end():
            pyautogui.scroll(-1)
        #event.clear()
        