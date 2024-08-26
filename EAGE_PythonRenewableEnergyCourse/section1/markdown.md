# Markdown and Latex

**Objetives**
* Provide an *overview* of Markdown language.
---

## Introduction

[Markdown](https://www.markdownguide.org/) is a lightweight markup language that you can use to add formatting elements to plaintext text documents. It is a fast and easy way to take notes, create content for a website, and produce print-ready documents, created by John Gruber in 2004.

You are probably familiar with [WYSIWYG](https://en.wikipedia.org/wiki/WYSIWYG) (What You See Is What You Get) editors, such as Microsoft Word or Google Docs. In this type of editors you see immediately what you changed in the document, but you don't see the program instructions in the form of the source code. When you create a Markdown-formatted file, you add Markdown syntax to the text to indicate which words and phrases should look different. 

In order to add format into the text, you use the *Markdown* syntax. For example, to denote a heading, you add a number sign before it (e.g., # Heading One). Or to make a phrase bold, you add two asterisks before and after it (e.g., **this text is bold**). It may take a while to get used to seeing Markdown syntax in your text, especially if you’re accustomed to WYSIWYG applications.

## Why Markdown?

Despite the fact that you can use Markdown in many different ways, the most common way to use Markdown is to create a document that is intended to be read by a computer. For example, in *WhatsApp* you can format the text using:
* Italic:  ```_text_```
* Bold: ```*text*```
* Strikethrough: ~text~
* Monospace: 
```markdown
```text```
```

As we noticed from the previous section, Google Colab support two types of cells: *text* and *code*. Text cells are formatted using a Markdown. To see the Markdown source, double-click a text cell, showing both the Markdown source and the rendered version. Above the Markdown source there is a toolbar to assist editing.

## Anatomy of a Markdown file

### Headers

```markdown
# H1 (Title)

## H2 (Subtitle)

### H3 (Subsubtitle)

#### H4 (Subsubsubtitle)

##### H5 (Subsubsubsubtitle)

###### H6 (Subsubsubsubsubtitle)

```
### Formatting
* Italic:  ```*text*_*```
* Bold: ```**text**```
* Strikethrough: ~text~
* Monospace: 
```markdown
```code```
```

### Links
```markdown
[Link text Here](https://link-url-here.org)
```

### Images
```markdown
![imageName](imageLocation.png)
```

### Tables
```markdown
First column name  | Second column name 
-------------------|------------------
Row 1, Col 1       | Row 1, Col 2 
Row 2, Col 1       | Row 2, Col 2 
```

A very useful tool is a [Markdown table generator](https://www.tablesgenerator.com/markdown_tables).

### Unordered Lists
```markdown
* One
* Two
* Three
```
### Ordered Lists
```markdown
1. One
2. Two
3. Three
```

### Horizontal Line
```markdown
---
```

### Color Font
```markdown
<font color=blue|red|green|pink|yellow>Text</font>
```

* <font color=red>Red</font>
* <font color=blue>Blue</font>
* <font color=green>Green</font>
* <font color=pink>Pink</font>
* <font color=yellow>Yellow</font>

### Mathematical Expressions
To embed a $\LaTeX$ equation within cell text in Google Colab (or Jupyer Notebook) we just need to enclosed the equation in dollar signs (`$`). For looking up symbols you may need, you can use any of the many [cheat sheets](http://users.dickinson.edu/~richesod/latex/latexcheatsheet.pdf) online. A very useful tool to generate Latex equation is A very useful tool is an online [$\LaTeX$ equation generator](https://latex.codecogs.com/eqneditor/editor.php).

#### Latex

[$\LaTeX$](https://www.latex-project.org/) is a document preparation system for high-quality typesetting. It is most often used for medium-to-large technical or scientific documents but it can be used for almost any form of publishing. Notice that LaTeX is not a word processor! Instead, LaTeX encourages authors not to worry too much about the appearance of their documents but to concentrate on getting the right content. 

The equation of a line is $y = a*cos(\Delta) + b^{-t+cos(\alpha)}$

But if we now put double `$$` at the beginning and at the end of our expression, we get:

$$A = \frac{cos(\theta)}{sin(2*\alpha)}$$

or something more complicated, like:

$$\int_{a}^{b}\lim_{x}\frac{\partial^2 }{\partial x^2}$$