# Genesis labs

---
## Create Python environment

Using venv (built-in, simplest)
```
#!/bin/bash -f
python -m venv myenv               # Create
source myenv/bin/activate          # Activate
pip install -r requirements.txt    # Install required packages
deactivate                         # Deactivate
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

<video width="550" height="400" alt="image" src="https://github.com/user-attachments/assets/d4453114-41d3-4ec3-b328-85f41017c13a"/>

<img width="550" height="400" alt="image" src="https://github.com/user-attachments/assets/d5f2cee3-2a03-40bd-8750-db07045d7915" />

---
## Labs: Car simulation

```
#!/bin/bash -f
source ./myenv/bin/activate
python3 -m carsim.test
```
