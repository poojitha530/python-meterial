#!/usr/bin/env python
# coding: utf-8

# ## Functions in Python
#  The function is a block of related statements that performs a specific task when it is called.
# 
# Functions helps in breaking our program into smaller and modular chunks which makes our program more organized and manageable. Also, it avoids repetition and makes the code reusable.
# 
# Functions in python can be divided into two types
# 
# Built-in Functions : Functions that are built into Python
# 
# User-defined Functions : Functions defined by the users themselves.
# 

# In[1]:


def square(n):
    return n**2
square(2)


# In[2]:


print(square(4))


# In[3]:


square(5)


# ##### the pass Statement
# The pass statement is used to avoid getting an error in empty functions as function definitions can not be empty in python.

# In[4]:


def smile():
    pass


# ##### Docstring
# The first string after the function header is called the docstring and is short for documentation string.

# In[5]:


def docstring():
    """this is documentation"""
print(docstring.__doc__)


# ##### Default Parameters
# If we call the function without argument, it uses the default value.

# In[11]:


def add(a=10,b=20):
    print("sum of a=",a,"and b=",b ,"is", a+b)
add()
#calling function without any parameter
#a and b will be default


# In[12]:


#calling function with only one parameter
#a=30 and b will be default
add(30)


# In[13]:


#calling function with only one parameter
#b=50 and a will be default
add(b=50)


# In[14]:


#calling function with  both parameters
#a=100 and b =300
add(100,300)


# ##### Keyword Parameters
# To allow the caller to specify the argument name with values so that caller does not need to remember the order of parameters.
# 

# In[16]:


#function
def students(first_name,last_name,rank):
    print("hello",first_name,last_name,"your rank is",rank)


# In[18]:


students(first_name="pujitha",last_name="sri",rank=1)


# In[19]:


students(last_name="sri",first_name="pooja",rank=4)


# In[20]:


students(rank=2,last_name="sri",first_name="pooja")


# ##### Arbitrary Parameters
# If we are not sure with number of arguments that will be passed in our function, we can use * before the parameter name so that the function will receive a tuple of arguments, and we can access the items accordingly.

# In[24]:


def welcome(*names):
    print("\n total names are:",len(names))
    for i in names:
        print("welcome to keats &kcp",i)


# In[25]:


welcome("pooja","keziya","sureka","rani","mouni","rahelu")


# In[31]:


def welcome(*names):
    for i in names:
        print("welcome to keats &kcp",i)
    print("\n total number of names:",len(names))


# In[32]:


welcome("pooja","keziya","sureka","rani","mouni","rahelu")


# In[42]:





# In[43]:


def details( **name ):
    print("First Name :" + name["fname"])

# Passing 2 parameters
details(fname="Muskan", lname="Agarwal")

# Passing 1 parameter
details(fname="Anchal")


# In[ ]:




