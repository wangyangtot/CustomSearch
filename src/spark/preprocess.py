from warcio.archiveiterator import ArchiveIterator

from langdetect import detect
import justext



class preprocessor ( object ) :
    def __init__(self):
        self.DESIRED_LANGUAGE = 'en'
        self.self.ALLOWED_CONTENT_TYPES = {"application/http; msgtype=response" , "message/http"}

    def ignoreWARCRecord(self,record) :
        # print ( record.rec_type )
        # print ( record.content_type )
        print ( record.length )

        if record.rec_type != 'response' :
            return True
        if record.length >= 10000000 :
            return True

        if not record.content_type or record.content_type not in self.ALLOWED_CONTENT_TYPES :
            return True

        return False



    def removeBolerplate(self,html) :


        try :
            paragraphs = justext.justext ( html , justext.get_stoplist ( "English" ) )
            plainText = [ ]
            for paragraph in paragraphs :
                if not paragraph.is_boilerplate :
                    plainText.append ( paragraph.text )
            return "".join ( plainText )

        except :
            return ""



    def isDesiredLanguage(self,plainText , DESIRED_LANGUAGE) :
        try :
            pre_language = detect ( plainText )
        except :
            return False
            # if pre_language != 'en' :
            #   print ( pre_language )
        return pre_language == self.DESIRED_LANGUAGE


    def getWARCRecord(self,path_to_files) :
        with open ( path_to_files , 'rb' ) as stream :
            for record in ArchiveIterator ( stream ) :
                if not self.ignoreWARCRecord ( record ) :
                    html = record.raw_stream.read ( )
                    if html :
                        plainText = self.removeBolerplate ( html )
                        if plainText and isDesiredLanguage ( plainText , DESIRED_LANGUAGE ):
                            print(plainText)



if __name__ == '__main__' :
    path_to_files = '/home/ubuntu/cc-pyspark/crawl-data/CC-MAIN-2017-13/segments/1490218186353.38/warc/' \
                    'CC-MAIN-20170322212946-00000-ip-10-233-31-227.ec2.internal.warc.gz'
    pre=preprocessor()
    pre.getWARCRecord ( path_to_files )
