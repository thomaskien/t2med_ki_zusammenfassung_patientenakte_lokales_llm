KI-Analyse der Patientenakte T2med mit automatischem Import über GDT


* lm studio muss installiert sein ->
* lms server start
<pre>
python3 -m venv .venv
source .venv/bin/activate
pip install watchdog pypdf openai
python ~/pdf_ki_watch.py
</pre>

* geräteconfig im screenshot
* beispieldateien: input-pdf, output als markdown, output als GDT

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Igjl7aTUNco/0.jpg)](https://www.youtube.com/watch?v=Igjl7aTUNco)
