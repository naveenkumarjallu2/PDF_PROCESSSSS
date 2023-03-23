# import fitz # PyMuPDF
# import io
# from PIL import Image
# # file path you want to extract images from
# file = r"D:\project_pdf_profile\compelete_pdf_editor_model\static\Full_tender_doc_IRCON.pdf"
# # open the file
# pdf_file = fitz.open(file)
# srt=int(input('enter start page: '))
# end=int(input('enter end page: '))
# # iterate over PDF pages
# for page_index in range(srt,end+1):
#     # get the page itself
#     page = pdf_file[page_index]
#     # get image list
#     image_list = page.get_images()
#     # printing number of images found in this page
#     if image_list:
#         print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
#     else:
#         print("[!] No images found on page", page_index)
#     for image_index, img in enumerate(image_list, start=1):
#         print('img',img)
#         # get the XREF of the image
#         xref = img[0]
#
#         # extract the image bytes
#         base_image = pdf_file.extract_image(xref)
#         print("base_image",base_image)
#         b=base_image['xres']=100
#         image_bytes = base_image["image"]
#         # get the image extension
#         image_ext = base_image["ext"]
#         # load it to PIL
#         image = Image.open(io.BytesIO(image_bytes))
#         print('image',image)
#         # save it to local disk
#         image.save(open(f"static\image{page_index + 1}_{image_index}.{image_ext}", "wb"))

# Import the Images module from pillow
from PIL import Image

# Open the image by specifying the image path.
image_path = r"D:\imgs\naveen\1.jpg"
image_file = Image.open(image_path)

# the default
image_file.save("image_name_d.jpg", quality=95)

# Changing the image resolution using quality parameter
# Example-1
image_file.save("image_name_ex1.jpg", quality=25)

# Example-2
image_file.save("image_name_ex2.jpg", quality=1)
