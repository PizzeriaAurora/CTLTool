from RefCheck.propgen import PropertyGenerator
if __name__ == "__main__":
    try:
        # Define a larger set of atomic propositions for better partitioning
        props = ["critical", "request", "grant", "error", "busy", "ready", "done", "waiting"]
        num_classes = 3
        
        prop_gen = PropertyGenerator(
            atomic_props=props,
            num_equivalence_classes=num_classes
        )

        print(f"--- Property Generator Initialized ---")
        print(f"Atomic Propositions: {props}")
        print(f"Divided into {num_classes} equivalence classes:\n")
        for i, class_props in enumerate(prop_gen.prop_classes):
            print(f"  Class {i}: {class_props}")
        print("-" * 40)
        
        # Generate 1 base property per class, each with 2 refinements
        generated_properties, a, b = prop_gen.generate(num_per_class=1, num_refinements=2)

        for class_idx, prop_list in generated_properties.items():
            class_atoms = prop_gen.prop_classes[class_idx]
            print(f"\n--- Properties for Class {class_idx} (Atoms: {class_atoms}) ---\n")
            
            # The first property is the base, the rest are refinements
            print(f"  Base Property:")
            print(f"    Custom: {prop_list[0]['custom']}")
            print(f"    NuSMV:  {prop_list[0]['nusmv']}\n")

            if len(prop_list) > 1:
                print(f"  Refinements:")
                for i, prop_pair in enumerate(prop_list[1:]):
                    print(f"    {i+1}. Custom: {prop_pair['custom']}")
                    print(f"       NuSMV:  {prop_pair['nusmv']}\n")
        print(a)
        print(b)
    except ValueError as e:
        print(f"Error: {e}")
#
#def generate(self, num_base: int, num_refinements: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
#        """
#        Generates properties, returning both custom and NuSMV syntax.
#
#        Returns:
#            A tuple containing (all_properties, base_properties).
#            Each list contains dicts: [{'custom': str, 'nusmv': str}, ...].
#        """
#        base_props = {}
#        base_props["custom"] = []
#        base_props["nusmv"] = []
#        for _ in range(num_base):
#            custom_p = self.generate_base_property()
#            nusmv_p = to_nusmv_syntax(custom_p)
#            base_props["custom"] += [custom_p]
#            base_props["nusmv"] += [nusmv_p]
#            #base_props.append({'custom': custom_p, 'nusmv': nusmv_p})
#        
#        all_props = base_props
#
#        for _ in range(num_refinements):
#            prop_to_refine = random.choice(all_props["custom"])
#            # Refine the custom syntax version
#            refined_custom = self.refine_property(prop_to_refine)
#            # Translate the new refined version
#            refined_nusmv = to_nusmv_syntax(refined_custom)
#            base_props["custom"] += [refined_custom]
#            base_props["nusmv"] += [refined_nusmv]
#            #all_props.append({'custom': refined_custom, 'nusmv': refined_nusmv})
#        
#        #random.shuffle(all_props)
#        return all_props, base_props