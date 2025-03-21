import time
from colorama import Fore, Style, init

# Initialize colorama (for Windows compatibility)
init(autoreset=True)

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        # Display with colors and styles
        print(f"{Style.BRIGHT}{Fore.YELLOW}Function '{func.__name__}'{Fore.RESET} "
              f"{Fore.GREEN}executed in {Fore.RED}{execution_time:.4f} {Fore.GREEN}seconds{Style.RESET_ALL}")
        
        return result
    return wrapper
