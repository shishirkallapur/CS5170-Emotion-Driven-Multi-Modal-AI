from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, WebBaseLoader



class DocLoaders:
    
    def __init__(self):
        pass
        
        
        
    def pdf_loader(self, file_path):
        return PyPDFLoader(file_path).load()
    
    def txt_loader(self, file_path):
        return TextLoader(file_path, encoding='utf-8').load()
    
    def web_loader(self, file_path):
        return WebBaseLoader(file_path).load()
    
    def run(self, file_path):
        # setup different loaders
        if ".pdf" in file_path:
            return self.pdf_loader(file_path)
        elif ".txt" in file_path or ".md" in file_path:
            return self.txt_loader(file_path)
        elif "www" in file_path or "http" in file_path:
            return self.web_loader(file_path)
    
    

        