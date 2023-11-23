from torch.utils.data import Dataset


class BackdoorImageDataset(Dataset):
    def __init__(self, nav_db):
        self.nav_db = nav_db

    def __len__(self):
        return len(self.nav_db.scanvp_refer)
    
    def __getitem__(self, i):
        scan, viewpoint = self.nav_db.scanvp_refer[i]
        inputs = self.nav_db.get_input(scan, viewpoint)
        output = {}
        output["ob_pano_images"] = inputs["ob_pano_images"]
        output["ob_img_fts"] = inputs["ob_img_fts"]
        output["stop_ft"] = inputs["stop_ft"]

        return output