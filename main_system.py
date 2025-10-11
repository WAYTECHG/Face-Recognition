import os

if __name__ == "__main__":
    while True:
        print()
        print("-"*120)
        print("\t\t\tWELCOME TO FACE RECOGNITION SYSTEM")
        print("-"*120)
        print("1. EigenFace Recognition system in Scratch")
        print("2. Improved EigenFace Recogntion system using sklearn (Able to Identified Unknown Person)")
        print("3. FisherFace Recognition System (Well performance on identifying Unknown Person)")
        print("4. Exit")
        print("-"*120)
        n = eval(input("Enter your face recognition system number: "))
        print("-"*120)

        if n == 1:
            print()
            os.system('python eigenface_scratch.py')

        elif n == 2:
            print()
            os.system('python eigenface_sklearn.py')

        elif n == 3:
            print()
            os.system('python fisherface_sklearn.py')

        elif n == 4:
            print("Thank you for using the system")
            print("-"*80)
            break
        else:
            print("Invalid Choice")


