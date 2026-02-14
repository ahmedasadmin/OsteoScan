import argparse 
parser = argparse.ArgumentParser()
parser.add_argument(
                
                    "-v",
                    "--verbose", 
                    help="increase output verbosity",
                    type=int,
                    choices = [0, 1, 2]
                )
parser.add_argument(
                    "square",
                    type=int,
                    help="display a square of given number"
                )
args = parser.parse_args()
answer = args.square**2

if args.verbose == 2:
        print(f"The square of {args.square} equals {answer}") 
elif args.verbose == 1:
        print(f"{args.square}^2 = {answer}") 
else: 
        print(answer) 

