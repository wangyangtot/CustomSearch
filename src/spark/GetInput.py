from warcio.archiveiterator import ArchiveIterator

from langdetect import detect
import justext
DESIRED_LANGUAGE='en'
ALLOWED_CONTENT_TYPES = {"text/html" , "application/xhtml+xml"}



def getWARCRecord(path_to_files) :
    with open ( path_to_files , 'rb' ) as stream :
        for record in ArchiveIterator ( stream ) :
            if not ignoreWARCRecord ( record ) and isDesiredLanguage(record) :
                plainText=removeBolerplate ( record.raw_stream)
                header=record.ec_headers
                print(plainText)

def removeBolerplate(html):
    paragraphs=justext.justext ( html , justext.get_stoplist ( "English" ) )
    plainText=[]
    for paragraph in paragraphs :
        if not paragraph.is_boilerplate :
            plainText.append(paragraph.text)
    return plainText

def isDesiredLanguage(record,DESIRED_LANGUAGE):
    return detect(record.content_stream)==DESIRED_LANGUAGE

def ignoreWARCRecord(record) :
    if record.rec_type != 'response' :
        return True
    elif record.length >= 10000000 :
        return True
    elif not record.content_type or record.content_type not in ALLOWED_CONTENT_TYPES :
        return True

    return False
if __name__ == '__main__':
    getWARCRecord(arg[0])
