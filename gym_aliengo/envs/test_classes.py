# class Person:
#   def __init__(self, fname, lname):
#     self.firstname = fname
#     self.lastname = lname

#   def printname(self):
#     print(self.firstname, self.lastname)

# class Student(Person):
#   def __init__(self, fname, lname, year):
#     super().__init__(fname, lname)
#     self.graduationyear = year

#   def welcome(self):
#     print("Welcome", self.firstname, self.lastname, "to the class of", self.graduationyear)

#   def printname(self):
#     super().printname()

# if __name__=="__main__":
#     student = Student('jeremiah','coholich', '2019')
#     student.printname()


class A:
  def __init__(self): self.foo()     
  def foo(self): print('A.foo')      
  def bar(self): self.foo()                  

class B(A):      
  def foo(self):
    super().foo()
    super().fuckmeup()
    print('Child foo has been called')    

if __name__ == '__main__': 
  b = B()