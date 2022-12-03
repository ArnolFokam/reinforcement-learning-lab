<div align="center">
<img src="https://hotemoji.com/images/dl/5/microscope-emoji-by-twitter.png" alt="emoji" style="width: 128px; height: 128px"></img>
</div>


# The Reinforcement Learning Lab

## Why this project?

I like styding various fields of mathematics and computer science, with a particular bias towards artificially intelligent algorithms. Therefore it would be nice to store all the knowledge I have gathered so far during my learning journey.

## What is included in this project?

- ✍️ Personal [notes](/notes) from [books](/notes/books) and [papers](/notes/papers) read. I recommend reading them through [obsidian](https://obsidian.md/) to easily navigate through the notes.
- The Python library 📚 for this project [rl_lab](/rl_lab/) that contains implementation of  different algorithms learnt and re-usable components to perform reduntant tasks such as plotting graphs and training agents.
- Python [notebooks](/notebooks) 📓 that demonstrates the concepts learned with visualizations included as much as possible.

## Set Up Environment

There are certain steps you need to do to be able to run the notebooks.

Assuming you have a suitable [pip](https://docs.python.org/3/library/venv.html) or [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) environment at your disposal, please run the following commands.

```bash
pip install -r requirements.txt
pip install -e .
pre-commit install
```

And that's it. Now you can run all the notebooks of this project.

## Usage

If you want to use the functions from the library, you can easily do that by importing the library in your python file or notebook.

Here is the code to initialize a bandit algorithm and its algorithm.

```python
from rl_lab.agents.bandits import Jumper
from rl_lab.environments.bandits import LineWalkEnvironment

allowed_jumps = [-5, -1, 1, 5]
timesteps = 40
line_length = 256

environment = LineWalkEnvironment(line_legnth)
agent = Jumper(allowed_jumps, timesteps, environment)
```

## Contributing

Since these are my personal notes, I do not expect any addition of new algorithms or concepts not touched in the notes. However, feel free to raise an issue if you spotted an error somewhere or just wish to discuss a particular concept implemented in this project.
