# thermal-defect-detection



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.com/skylarkdrones/spectra/compute-services/thermal-defect-detection.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

For running the docker file, build the image first - 
```
docker build -t "image_name" .
```

Once docker image is ready, to run it on different orthos - 

```
docker run docker run -v "path_of_ortho:/app/data/ortho/ortho.tif" -v "path_of_where_to_save:/app/output" solar_panel
```
For example - 

```
docker run -v "C:\Users\Skylark\Desktop\Solar_AI\thermal-defect-detection\data\ortho\Envoler_Solar_150MW_thermal.tif:/app/data/ortho/ortho.tif" -v "C:\Users\Skylark\Desktop\Solar_AI:/app/output" solar_panel
```