import os

class color:
    def __init__(self):
        try:
            from colorama import Fore, Style
            print("Colorama Imported Successfully")
        except:
            print ("Coloram not found, Installing ........\n ")
            os.system("conda install -y colorama")
            import colorama
        finally:

            print("Cloroama initialized\n Running the Engine...")
            self._run()
            

    def _run(self):
        
        os.system("mode con: cols=120 lines=30")
        self.RED = self._RED()
        self.BLUE = self._BLUE()
        self.GREEN = self._GREEN()
        self.YELLOW = self._YELLOW()
        self.DIM = self._DIM()
        self.BRIGHT = self._BRIGHT()
        self.RESET = self._RESET()        
    
    def _RED(self):
        return Fore.RED
    def _BLUE(self):
        return Fore.BLUE
    def _GREEN(self):
        return Fore.GREEN
    def _YELLOW(self):
        return Fore.YELLOW
    def _DIM(self):
        return Style.DIM
    def _BRIGHT(self):
        return Style.BRIGHT
    def _RESET(self):
        return Style.RESET_ALL

'''
print(Fore.BLUE + Style.BRIGHT + "This is the color of the sky" + Style.RESET_ALL)
print(Fore.GREEN + "This is the color of grass" + Style.RESET_ALL)
print(Fore.BLUE + Style.DIM + "This is a dimmer version of the sky" + Style.RESET_ALL)
print(Fore.YELLOW + "This is the color of the sun" + Style.RESET_ALL)
'''