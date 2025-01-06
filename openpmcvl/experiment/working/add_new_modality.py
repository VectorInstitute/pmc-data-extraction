from mmlearn.datasets.core import Modalities, find_matching_indices
from mmlearn.datasets.core.modalities import Modality

if __name__ == "__main__":
    # print modalities
    print(f"Modalities: {Modalities}")
    print(f"Modalities.list_modalities: {Modalities.list_modalities()}")

    # get a modality
    mod = "rgb"
    modal = Modalities.get_modality(mod)
    print(f"modal: {modal}")
    print(f"modal.properties: {modal.properties}")
    print(f"Modalities.get_modality_properties(mod): {Modalities.get_modality_properties(mod)}")

    # add a new modality
    newmodname = "patient_q"
    Modalities.register_modality(name=newmodname)
    print(f"Modalities.list_modalities: {Modalities.list_modalities()}")
    print(f"Modalities.get_modality_properties(newmodname): {Modalities.get_modality_properties(newmodname)}")

    # add another modality
    newmodname = "patient_t"
    Modalities.register_modality(name=newmodname)
    print(f"Modalities.list_modalities: {Modalities.list_modalities()}")
    print(f"Modalities.get_modality_properties(newmodname): {Modalities.get_modality_properties(newmodname)}")

    # test attribute as property
    print(f"Modalities.RGB: {Modalities.RGB}")
    print(f"Modalities.PATIENT_Q: {Modalities.PATIENT_Q}")


