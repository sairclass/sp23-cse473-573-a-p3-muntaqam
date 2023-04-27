# Installation guide

**Project page:**
[https://pypi.org/project/face-recognition/](https://pypi.org/project/face-recognition/)

**Home page for face\_recognition library:**
[https://github.com/ageitgey/face\_recognition](https://github.com/ageitgey/face_recognition)

## Linux and MacOS

[https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)

## Windows

This library is not officially supported on Windows operating systems. Although there is a way to get it working on Windows: [https://github.com/ageitgey/face\_recognition/issues/175#issue-257710508](https://github.com/ageitgey/face_recognition/issues/175#issue-257710508)

One other option(for the latest Windows) is to use Windows subsystem Linux([WSL](https://learn.microsoft.com/en-us/windows/wsl/install)) with your choice of Linux distribution(Ubuntu - simplest of the bunch) and follow the Linux setup there.

## Conda

With Conda, you can try this setup:
[https://github.com/ageitgey/face\_recognition/issues/175#issuecomment-636326300](https://github.com/ageitgey/face_recognition/issues/175#issuecomment-636326300)

### Prerequisites

This library uses dlib, a modern C++ toolkit containing machine learning algorithms and tools, to perform face recognition. 

dlib and CMake are required. Please make sure your C compilers are in compatible versions for CMake that you are installing.

Following the steps from discussions on this GitHub repository should help you debug issues during the setup. **We highly recommend going through the installation guide on the GitHub/project page links.**
