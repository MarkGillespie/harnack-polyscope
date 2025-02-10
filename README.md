# Harnack tracing experiments

This Rep contains some C++ experiments from our [Ray Tracing Harmonic Functions](https://markjgillespie.com/Research/harnack-tracing/index.html) paper. It was mostly used to debug code that I wrote for our [Blender implementation](https://github.com/MarkGillespie/harnack-blender), but may be useful on its own. But be warned: this code is messy research code hacked together while running experiments on a deadline. So feel free to reach out if you have problems using it.

![Screenshot of points sampled on a nonplanar polygon in ourUI](images/example.jpt)

## Building and Running
On mac/linux, you can set up this project with the following commands.
```bash
git clone --recursive https://github.com/MarkGillespie/harnack-polyscope.git
cd harnack-polyscope
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make -j7
bin/run
```

Running the program with no arguments will load up a collection of simple nonplanar polygons used in our experiments. You can also pass in an `obj` file to load its faces as (possibly-nonplanar) polygons, or pass in a polygon specified as a `*.loops` file, where each line lists out the xyz coordinates of each vertex in a loop, e.g.:
```
v1.x v1.y v1.z v2.x v2.y v2.z .... vk.x vk.y vk.z
```

### Parameters
By default, only the first face of an `obj` file or first loop in a `*.loops` file is intersected by Harnack tracing queries. You can adjust which face/loop is considered via `Parameters > loop_id`.

The number of rays which are shot can be modified through `Parameters > resolution x` and `Parameters > resolution y`.

In the UI there is a button to attempt to mesh the 2π solid angle level set of the input loops (`Sphere Tracing > Build Sphere Tracing Mesh`), but this function uses a lot of questionable heuristics, and was only ever used on very simple polygons. If you pass it a complicated input, it will probably produce garbage.
