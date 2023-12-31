install: model.py
		pip install -r requirements.txt

lint:
	pylint --disable=R,C model.py

push:
	git add . && git commit -m "Adding stuff" && git push