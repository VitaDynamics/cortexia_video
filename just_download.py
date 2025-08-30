import cortexia 

# predownload default caption model

caption_ft = cortexia.create_feature("caption")

caption_ft._initialize()

detection_ft = cortexia.create_feature("detection")

detection_ft._initialize()

listing_ft = cortexia.create_feature("listing")

listing_ft._initialize()

segmentation_ft = cortexia.create_feature("segmentation")

segmentation_ft._initialize()