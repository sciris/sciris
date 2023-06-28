---
title: 'Sciris: Simplifying scientific software in Python'

tags:
  - python
  - scientific software development
  - computational science
  - numerical utilities
  - containers
  - plotting

authors:
  - name: Cliff C. Kerr
    orcid: 0000-0003-2517-2354
    corresponding: true
    affiliation: "1, 2" 
  - name: Paula Sanz-Leon
    orcid: 0000-0002-1545-6380
    affiliation: "1"
  - name: Romesh G. Abeysuriya
    orcid: 0000-0002-9618-6457
    affiliation: "1, 3"
  - name: George L. Chadderdon
    orcid: 0000-0002-3034-2330
    affiliation: "3, 4"
  - name: Vlad-Ştefan Harbuz
    affiliation: 5
  - name: Parham Saidi
    affiliation: 5
  - name: Maria del Mar Quiroga
    orcid: 0000-0002-8943-2808
    affiliation: "3, 6"
  - name: Rowan Martin-Hughes
    orcid: 0000-0002-3724-2412
    affiliation: 3
  - name: Sherrie L. Kelly
    orcid: 0000-0002-6232-5586
    affiliation: 3
  - name: Jamie A. Cohen
    orcid: 0000-0002-8479-1860
    affiliation: 1
  - name: Robyn M. Stuart
    orcid: 0000-0001-6867-9265
    affiliation: "1, 7"
  - name: Anna Nachesa
    affiliation: 8

affiliations:
 - name: Institute for Disease Modeling, Global Health Division, Bill \& Melinda Gates Foundation, Seattle, USA
   index: 1
 - name: School of Physics, University of Sydney, Sydney, Australia
   index: 2
 - name: Burnet Institute, Melbourne, Australia 
   index: 3
 - name: CAE USA, Tampa, USA
   index: 4
 - name: Saffron Software, Bucharest, Romania
   index: 5
 - name: Melbourne Data Analytics Platform, The University of Melbourne, Melbourne, Australia
   index: 6
 - name: Department of Mathematical Sciences, University of Copenhagen, Copenhagen, Denmark
   index: 7
 - name: Google, Zürich, Switzerland
   index: 8

date: 30 March 2023
bibliography: paper.bib

---

# Summary 

[Sciris](http://sciris.org) aims to streamline the development of scientific software by making it easier to perform common tasks. Sciris provides classes and functions that simplify access to frequently used low-level functionality in the core libraries of the scientific Python ecosystem (such as `numpy` for math and `matplotlib` for plotting), as well as in libraries of broader scope (such as `multiprocess` for parallelization and `pickle` for saving and loading objects). While low-level functionality is valuable for developing robust software applications, it can divert focus from the scientific problems being solved. Some of Sciris' key features include: ensuring consistent dictionary, list, and array types (e.g., enabling users to provide inputs as either lists or arrays); enabling ordered dictionary elements to be referenced by index; simplifying datetime arithmetic by allowing date input in multiple formats, including strings; simplifying the saving and loading of files and complex objects; and simplifying the parallel execution of code. With Sciris, users can often achieve the same functionality with fewer lines of code, avoid reinventing the wheel, and spend less time looking up recipes on Stack Overflow. This can make writing scientific code in Python faster, more pleasant, and more accessible, especially for people without extensive training in software development.


# Statement of need

## The landscape of scientific software
With the increasing availability of large volumes of data and computing resources, scientists across multiple fields of research have been able to tackle increasingly complex problems. But to harness these resources, the need for domain-specific software has become much greater. As the complexity of the questions being tackled has increased, so too has the amount of code used to answer them, creating a steep learning curve and significant burden of code review [@burden-codereview]. For some scientists, this increasing reliance on software has created a barrier between themselves and the science they want to do. It is these people – people who want things to "just work" rather than worry about the implementation details – who are the primary audience for Sciris. (In contrast, people who care a lot about implementation details – such as those who love using type hints – will likely *not* find Sciris to be as helpful.)

Scientific code workflows (e.g., either a full cycle in the development of a new software library, or in the execution of a one-off analysis) typically rely on multiple codebases, including but not limited to: low-level libraries, domain-specific open-source software, and self-developed and/or inherited Swiss-Army-knife toolboxes (whose original developer may or may not be around to pass on undocumented wisdom). Several scientific communities have adopted collaborative, community-driven, open-source software approaches due to the significant savings in development costs and increases in code quality that they afford, such as [`astropy`](https://www.astropy.org/) [@robitaille2013astropy], [`fmriprep`](https://fmriprep.org) [@esteban2019fmriprep], and [`nextstrain`](https://nextstrain.org) [@hadfield2018nextstrain]. Despite this progress, a large fraction of scientific software development efforts remain a solo adventure [@kerr2019epidemiology]. This leads to proliferation of tools where resources are largely spent reinventing wheels of variable quality, which jeopardizes the code's minimum requirements of being "re-runnable, repeatable, reproducible, reusable, and replicable" [@benureau2018re].

In addition, low-level programming abstractions can make it harder to clarify the science. For instance, one of the reasons PyTorch has become popular in academic and research environments is its success in making models easier to write compared to TensorFlow [@pytorch-research]. The need for libraries that provide "simplifying interfaces" for research applications is reflected in the development of multiple libraries in scientific Python ecosystems that have enabled researchers to focus their time and efforts on solving problems, prototyping solutions, deploying applications, and educating their communities. In addition to PyTorch (simplifying/extending Tensorflow), other examples include seaborn (simplifying/extending Matplotlib) [@waskom2021seaborn], pingouin (simplifying/extending pandas), and PyVista (simplifying/extending VTK) [@sullivan2019pyvista], among many others. Sciris adds to this ecosystem as a "library of the gaps", addressing annoyances that are too small-scale to each need a dedicated library of their own, but common enough that together they add up to significant coding burden.

## Sciris in practice
The name [Sciris](https://github.com/sciris/sciris) is a portmanteau of "scientific" and "iris" (a reference to seeing clearly, as well as the Greek word for "rainbow"). We began work on it in 2014, initially to support development of [Optima HIV](https://github.com/optimamodel/optima) [@kerr2015optima; @kerr2020optima]. We repeatedly encountered the same inconveniences while building scientific webapps, and so we began collecting the tools we used to overcome them into a shared library. While Python is considered an easy-to-use language for beginners, the motivation that shaped Sciris' evolution was to further lower the barriers to accessing the numerous supporting libraries we were using.

Our investments in Sciris paid off when in early 2020 its combination of brevity and simplicity proved crucial in enabling the rapid development of the [Covasim](https://covasim.org) model of COVID-19 transmission [@kerr2021covasim]. Covasim's relative simplicity and readability, based in large part on its heavy use of Sciris, enabled it to become one of the most widely adopted models of COVID-19, used by students, researchers, and policymakers in over 30 countries [@kerr2022python].

In addition to Optima HIV and Covasim, Sciris is currently used by many other scientific software tools, such as [Optima Nutrition](https://github.com/optimamodel/nutrition) [@pearson2018optima], the [Cascade Analysis Tool](https://cascade.tools) [@kedziora2019cascade], [Atomica](https://atomica.tools) [@atomica], Optima TB [@gosce2021optima], the [Health Interventions Prioritization Tool](http://hiptool.org) [@fraser2021using], [SynthPops](https://synthpops.org) [@synthpops], and [FPsim](https://fpsim.org) [@o2022fpsim].

We believe using Sciris can lead to more efficient scientific code production for solo developers and teams alike, including increased longevity of new scientific libraries [@perkel2020challenge]. Some of the key functional aspects that Sciris provides are: (i) brevity through simple interfaces; (ii) "dejargonification"; (iii) fine-grained exception handling; and (iv) version management. We expand on each of these below, but first provide a vignette that illustrates many of Sciris' features.


# Vignette

Compared with a domain-specific language like MATLAB, even relatively simple scientific code in Python can require significant boilerplate. This extra code can obscure the key logic of the scientific question being addressed.

For example, imagine that we wish to sample random numbers from a user-defined function with varying noise levels, save the intermediate calculations, and plot the results. In vanilla Python, each of these operations is somewhat cumbersome. \autoref{fig:showcase-code} presents two functionally identical scripts; the one written with Sciris is considerably more readable and succinct. 

This vignette illustrates many of Sciris' most-used features, including timing, parallelization, feature-rich containers, file saving and loading, and plotting. For the lines of the script that differ, Sciris reduces the number of lines of code required from 33 to 7, a 79% decrease.

![Comparison of functionally identical scripts without Sciris (left) and with Sciris (right), showing a nearly five-fold reduction in lines of code required (excluding whitespace, comments, and the shared "wave generator" code), from 33 lines to 7. The resulting plots are shown in \autoref{fig:showcase-output}. \label{fig:showcase-code}](figures/sciris-showcase-code.png){ width=100% }

![Output of the scripts shown in \autoref{fig:showcase-code}, without Sciris (left) and with Sciris (right). The two plots are identical except for the new high-contrast colormap available in Sciris. \label{fig:showcase-output}](figures/sciris-showcase-output.png){ width=100% }


# Design philosophy

The aim of Sciris is to make common tasks simpler. Sciris includes implementations of heavily used code patterns and abstractions that facilitate the development and deployment of complex domain-specific scientific applications, and helps non-specialist audiences interact with these applications. We note that Sciris "stands on the shoulders of giants", and as such is not intended as a replacement of these libraries, but rather as an interface that facilitates a more effective and sustainable development process through the following principles:

*Brevity through simple interfaces*. Sciris packages common patterns requiring multiple lines of code into single, simple functions. With these functions one can succinctly express and execute frequent plotting tasks (e.g., `sc.commaticks`, `sc.dateformatter`, `sc.plot3d`); ensure consistent types, including containers (e.g., `sc.toarray`, `sc.mergedicts`, `sc.mergelists`), or even perform line-by-line performance profiling (`sc.profile`). Brevity is also achieved by extending functionality of well established objects (e.g., `OrderedDict` via `sc.odict`) and methods (e.g., `isinstance` via `sc.checktype` that enables the comparison of objects against higher-level types like `arraylike`), as well as wrapping useful third-party libraries (e.g., `pyyaml` via `sc.loadyaml`). In providing a curated collection of common data science tools, Sciris has similarities to R's [tidyverse](https://www.tidyverse.org/).

*Dejargonification*. Sciris aims to use plain function names (e.g., `sc.smooth`, `sc.findnearest`, `sc.safedivide`) so that the resulting code is as scientifically clear and human-readable as possible. Sciris also provides some [MATLAB](https://www.mathworks.com/products/matlab.html)-like functionality, and uses the same names (e.g., `sc.tic` and `sc.toc`; `sc.boxoff`) to minimize the learning curve for scientists who have MATLAB experience.

*Fine-grained exception handling*. Across many classes and functions, Sciris uses the keyword `die`, enabling users to set a locally scoped level of strictness in the handling of exceptions. If `die=False`, Sciris is more forgiving and softly handles exceptions by using its default (opinionated) behavior, such as printing a warning and returning `None` so users can decide how to proceed. If `die=True`, it directly raises the corresponding exception and message. This flexibility reduces the need for try-catch blocks, which can distract from the code's scientific logic.

*Version management*. Keeping track of dates, authors, and code versions, plus additional notes or comments, is an essential part of scientific projects. Sciris provides methods to easily save and load metadata to/from figure files, including Git information (`sc.savefig`, `sc.gitinfo`, `sc.loadmetadata`), as well as shortcuts for comparing module versions (`sc.compareversions`) or requiring them (`sc.require`).


# Examples of key features

Here we illustrate a smattering of key features in greater detail; further information on installation and usage can be found at [docs.sciris.org](https://docs.sciris.org). \autoref{fig:block-diagram} illustrates the functional modules of Sciris. Sciris is available on pip (`pip install sciris`).

![Block diagram of Sciris' functionality, grouped by high-level concepts and types of tasks that are commonly performed in scientific code.\label{fig:block-diagram}](figures/sciris-block-diagram-03.png){ width=100% }

## Feature-rich containers
One of the key features in Sciris is `sc.odict`, a flexible container representing an associative array with the best-of-all-worlds features of lists, dictionaries, and numeric arrays. This is based on `OrderedDict` from [`collections`](https://docs.python.org/3/library/collections.html), but supports list methods like integer indexing, key slicing, and item insertion:

```Python
data = sc.odict(a=[1,2,3], b=[4,5,6]) 
assert data['a'] == data[0]
assert data[:].sum() == 21
for i, key, value in data.enumitems():
  print(f'Item {i} is named "{key}" and has value {value}')
# Output:
# Item 0 is named "a" and has value [1, 2, 3]
# Item 1 is named "b" and has value [4, 5, 6]
```

## Numerical utilities
Indexing arrays is a common task in NumPy, but can be difficult due to incompatibilities of object type. `sc.findinds` will find matches even if two things are not exactly equal due to differences in type (e.g., floats vs. integers, lists vs. arrays). The code shown below produces the same result as calling `np.nonzero(np.isclose(arr, val))[0].`

```Python
sc.findinds([2,3,6,3], 3.0) 
# Output:
# array([1,3])
```

## Parallelization
A frequent hurdle scientists face is parallelization. Sciris provides `sc.parallelize`, which acts as a shortcut for using `multiprocess.Pool()`. By default it adjusts the pool size based on the CPUs available, but can also use either a fixed number of CPUs or allocate them dynamically based on load (`sc.loadbalancer`). This example shows three equivalent ways to iterate over multiple complex arguments:

```Python
def f(x, y):
   return x*y

out1 = sc.parallelize(func=f, iterarg=[(1,2),(2,3),(3,4)])
out2 = sc.parallelize(func=f, iterkwargs={'x':[1,2,3], 'y':[2,3,4]})
out3 = sc.parallelize(func=f, iterkwargs=[{'x':1, 'y':2}, 
                                         {'x':2, 'y':3}, 
                                         {'x':3, 'y':4}])
```

## Plotting
Numerous shortcuts for customizing and prettifying plots are available in Sciris. Several commonly used features are illustrated below, with the results shown in \autoref{fig:plotting-example}:

```Python
sc.options(font='Garamond') # Set custom font
x = sc.daterange('2022-06-01', '2022-12-31', as_date=True) # Create dates
y = sc.smooth(np.random.randn(len(x))**2)*1000 # Create smoothed random numbers
c = sc.vectocolor(y, cmap='turbo') # Set colors proportional to y values
plt.scatter(x, y, c=c) # Plot the data
sc.dateformatter() # Automatic x-axis date formatter
sc.commaticks() # Add commas to y-axis tick labels
sc.setylim() # Automatically set the y-axis limits
sc.boxoff() # Remove the top and right axis spines
```

![Example of plot customizations via Sciris, including x- and y-axis tick labels and the font.\label{fig:plotting-example}](figures/plotting-example.png){ width=70% }


# ScirisWeb

While a full description of [ScirisWeb](http://github.com/sciris/scirisweb) is beyond the scope of this paper, briefly, it builds on Sciris to enable the rapid development of Python-based webapps, including those powering [Covasim](https://app.covasim.org) and [Optima Nutrition](https://nutrition.optimamodel.com). By default, ScirisWeb uses [Vuejs](https://vuejs.org) and [sciris-js](https://github.com/sciris/sciris-js) for the frontend, [Flask](https://flask.palletsprojects.com) as the web framework, [Redis](https://redis.io) for the (optional) database, and Matplotlib/[mpld3](https://github.com/mpld3/mpld3) for plotting. However, ScirisWeb is completely modular, which means that it could also be used to (for example) link a [React](https://reactjs.org/) frontend to a [MySQL](https://www.mysql.com/) database with [Plotly](https://plotly.com/) figures. This modularity is in contrast to full-stack solutions such as [Shiny for Python](https://shiny.rstudio.com/py/), [Plotly Dash](https://github.com/plotly/dash), [Streamlit](https://streamlit.io), and [Voilà](https://voila.readthedocs.io). While these libraries are even easier to use than ScirisWeb (since they do not require any knowledge of JavaScript), they provide limited options for customization or switching between technology stacks. In contrast, ScirisWeb provides the flexibility of a custom-written webapp within the context of an "it just works" framework.


# Beyond Sciris

Like seaborn, Sciris aims to "facilitate rapid exploration and prototyping through named functions and opinionated defaults" [@waskom2021seaborn]. Eventually, a time may come when the user's opinions diverge from Sciris' defaults. Since most Sciris functions are standalone, individual functions can be replaced on as as-needed basis. For example, in situations where strictness is an asset (e.g., low-level libraries where an unexpected type is indicative of an error), the added flexibility that Sciris provides (e.g., the type-agnostic `sc.toarray`) can be a liability. As another example, `sc.odict` adds small but nonzero overhead to the `dict` built-in. While in most cases this performance difference is negligible (<500 ms per million set/get operations), for innermost loops of compute-intensive applications, `dict` should be used instead. Finally, since Sciris aims for breadth rather than depth, Sciris functions may eventually need to be supplanted by more powerful alternatives. For example, while `sc.parallelize` provides one-line parallelization on a local machine or single virtual machine, parallelizing across multiple machines requires more powerful libraries such as [Dask](https://www.dask.org/) [@rocklin2015dask], [Ray](https://www.ray.io/), or [Celery](https://docs.celeryq.dev/).


# Acknowledgements

The Sciris Development Team (info@sciris.org) wishes to thank David J. Kedziora, Dominic Delport, Kevin M. Jablonka, Meikang Wu, and Dina Mistry for providing helpful feedback on the Sciris library. David P. Wilson, William B. Lytton, and Daniel J. Klein provided in-kind support of Sciris development. Financial and personnel support has been provided by the United States Defense Advanced Research Projects Agency (DARPA) Contract N66001-10-C-2008 (2010–2014), World Bank Assignment 1045478 (2011–2015), the Australian National Health and Medical Research Council (NHMRC) Project Grant APP1086540 (2015–2017), the Australian Research Council (ARC) Discovery Early Career Research Award (DECRA) Fellowship Grant DE140101375 (2014–2019), Intellectual Ventures (2019–2021), and the Bill & Melinda Gates Foundation (2021–present).


# References
