while True:
    try:
        with open("Roy/Helper_Functions/Testing2.py", encoding="utf-8") as f:
            exec(f.read())
    except FileNotFoundError:
        print("File not found: Roy/Helper_Functions/Testing2.py")
    except Exception as e:
        print(f"An error occurred: {e}")