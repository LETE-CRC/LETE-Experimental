"""
===============================================================================
                      Python code for PIV analysis
   Created by Combustion Research Center CRC at LETE - Sao Paulo, Brasil
   Laboratory of Environmental and Thermal Engineering - LETE
   Escola Politecnica da USP - EPUSP
   
===============================================================================
version:2.0 - 05/2020: Helio Villanueva
"""

from termcolor import colored
import os

versions = ['version:0.0 - 02/2019: Helio Villanueva',
            'version:1.0 - 04/2019: Helio Villanueva',
            'version:1.1 - 08/2019: Helio Villanueva',
            'version:2.0 - 05/2020: Helio Villanueva']


htxt = '\n' + 79*'='
htxt += '\n                      Python code for PIV analysis'
htxt += '\n   Created by Combustion Research Center CRC at LETE - Sao Paulo, Brasil'
htxt += '\n   Laboratory of Environmental and Thermal Engineering - LETE'
htxt += '\n   Escola Politecnica da USP - EPUSP'
htxt += '\n\n' + versions[-1]
htxt += '\n' + 79*'=' + '\n'

etxt = '\n\n' + 79*'='
etxt += '\n END crcPIV\n'
etxt += 79*'=' + '\n'

def header():
    os.system('clear')
    print(colored(htxt,'green'))
    
    return 0

def end():
    print(colored(etxt,'green'))
    return 0