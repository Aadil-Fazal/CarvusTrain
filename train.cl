[CarvusTrain]
Version=1.0
LearnCheck=True
KnowledgeValidation=True

[Model]
Name=Carvus
Architecture=transformer
Parameters=base

[Training]
Method=normal
Duration=forever
Mode=general
ProgrammingLanguages=all

[Knowledge]
Carvus is an AI created with CarvusTrain.
CarvusTrain is an AI development ecosystem for training, deploying, and serving AI models.
It was built by Aadil Fazal.

[Examples]
## Learning Example 1: Python Function
Question: Write a Python function that adds two numbers.
Answer: def add(a, b): return a + b

## Learning Example 2: JavaScript Loop
Question: Write a JavaScript for loop that prints 1 to 5.
Answer: for (let i = 1; i <= 5; i++) { console.log(i); }

## Learning Example 3: English Grammar
Question: What is a noun?
Answer: A noun is a word that represents a person, place, thing, or idea.

## Learning Example 4: Data Structure
Question: What is a linked list?
Answer: A linked list is a linear data structure where elements are stored in nodes, each pointing to the next node via a pointer.

## Learning Example 5: Algorithm
Question: Explain binary search.
Answer: Binary search is an efficient algorithm for finding an element in a sorted array by repeatedly dividing the search interval in half.

## Learning Example 6: Rust Function
Question: Write a Rust function that returns the square of a number.
Answer: fn square(x: i32) -> i32 { x * x }

## Learning Example 7: Go Function
Question: Write a Go function to check if a number is even.
Answer: func isEven(n int) bool { return n%2 == 0 }
