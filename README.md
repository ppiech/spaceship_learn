



## Notes

### Editor 2-space indent

A more direct way to change the indentation is to directly edit the Jupyter config files in the .jupyter/nbconfig directory.:

notebook.json

```
{
  "CodeCell": {
      "cm_config": {
            "indentUnit": 2
	        }
		  }
}
```