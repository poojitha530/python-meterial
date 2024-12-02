#!/usr/bin/env python
# coding: utf-8

# In[3]:


from threading import Thread
from time import sleep


# In[1]:


class hello:
    def run(self):
        for i in range(5):
            print("hello")
class hi:
    def run(self):
        for i in range(5):
            print("hi")
            
            
t1=hello()
t2=hi()
t1.run()
t2.run()


# In[2]:


class hello(Thread):
    def run(self):
        for i in range(5):
            print("hello")
            sleep(1)
class hi(Thread):
    def run(self):
        for i in range(5):
            print("hi")
            sleep(1)
            
            
t1=hello()
t2=hi()
t1.start()
t2.start()
print("this is multiple threading in python")


# In[21]:


class hello(Thread):
    def run(self):
        for i in range(5):
            print("hello")
            sleep(1)
class hi(Thread):
    def run(self):
        for i in range(5):
            print("hi")
            sleep(1)
            
            
t1=hello()
t2=hi()
t1.start()
t2.start()
t1.join()
t2.join()
print("this is multiple threading in python")


# In[34]:


from threading import Thread
from time import sleep

class hello(Thread):
    def run1(self):
        for i in range(5):
            print("hello")
            sleep(1)
class hi(Thread):
    def run2(self):
        for i in range(5):
            print("hi")
            sleep(1)
            
            
t1=hello()
t2=hi()

t1.start()
sleep(0.2)
t2.start()

t1.join()
#sleep(1)
t2.join()
print("this is multiple threading in python")


# In[38]:


import threading
def print_numbers():
    for i in range(1,6):
        print(i)
thread= threading.Thread(target=print_numbers)

thread.start()
thread.join()


# In[ ]:




