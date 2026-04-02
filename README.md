# Genesis labs

---
## Create Python environment

Using venv (built-in, simplest)
```
#!/bin/bash -f
python -m venv myenv       # Create
source myenv/bin/activate  # Activate
deactivate                 # Deactivate
```

---
## Labs: Hello Genesis

```
#!/bin/bash -f
source ./myenv/bin/activate
python3 -m hello.lab2
python3 -m hello.lab3
python3 -m hello.lab4
python3 -m hello.lab5
```


---
## Labs: Car simulation

```
#!/bin/bash -f
source ./myenv/bin/activate
python3 -m carsim.test
```
