from jinja2 import Template
import os
from os import path


def create_pgfplot(filename):

    template = Template(r"""
    \documentclass{standalone}
    \pagestyle{empty}
    \usepackage[utf8]{inputenc}

    \usepackage{tikz}
    \usepackage{tikz-cd}
    \usepackage{pgfplots}
    \usepackage{layouts}
    \pgfplotsset{compat=1.14}

    \begin{document}
    \input{{filename}}
    \end{document}
    """)

    filename_temp = f"\u007b{filename}\u007d"
    latex_document = template.render(filename=filename_temp)

    filename = filename.replace(".pgf", ".tex")

    with open(filename, "w") as f:
        f.write(latex_document)

    # compile latex document
    os.system(f"pdflatex -interaction=nonstopmode {filename} >/dev/null")


if __name__ == "__main__":
    folders = [folder for folder in os.listdir() if path.isdir(folder)]
    folders.sort()
    for folder in folders:
        os.chdir(folder)
        # get all pgf files in current folder
        pgf_files = [file for file in os.listdir() if file.endswith(".pgf")]

        for file in pgf_files:
            create_pgfplot(file)

        # remove all .aux, .log and .tex files
        for file in os.listdir():
            if file.endswith(".aux") or file.endswith(".log") or file.endswith(".tex"):
                os.remove(file)
        print(f"Created pdfs for {folder}")
        os.chdir("..")
