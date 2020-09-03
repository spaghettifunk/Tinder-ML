This project was made for the course of Machine Learning at Uppsala University

`Authors`
Davide Berdin, Tobias Famulla

`Descritpion`

Inside `src` folder there is the `result` folder that contains the **csv** and **svg** files of our experiments

**How to start**

In folder webappflask you should put a folder called 'photos' structured in this way:

```
photos
|-- person1
|   |-- 1.jpg
|   |-- 2.jpg
|   |-- 3.jpg
|   |-- 4.jpg
|-- person2
|   |-- 1.jpg
|   |-- 2.jpg
|   |-- 3.jpg
|   |-- 4.jpg

[...]
```

After that do the following:

1. Create a new database using `create_db()` provided by `webappflask.models`
2. Create a model using `create_photo_db(gender=i) # where i = 1 (female), i = 2 (male)`
3. Photo in the library must be one gender (using the function twice on two different renamed folder and combine them afterwards)
4. Start the server using `runserver.py`
