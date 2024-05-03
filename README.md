# MORL

Use python 3.11

Install all the requirements

```bash
pip install -r requirements.txt
```

To run the experiment 

```bash
python experiments/flp/flp_exp_1__tabular.py
```


This will run the experiment for the first setting of the FLP problem. The print out the stats of each baseline in order (Goal only, Goal + cookie, Goal and cookie as separate Q-funcs (our method))

The Q-values are saved and you can view them in browser via

```./build.sh```

then open `index.html` in your browser.