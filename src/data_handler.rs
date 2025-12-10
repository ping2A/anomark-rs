use ahash::AHashMap;
use anyhow::Result;
use csv::ReaderBuilder;
use regex::Regex;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct DataRow {
    #[serde(flatten)]
    pub fields: AHashMap<String, String>,  // Now works with serde feature enabled
}

/// Replace SID patterns with placeholder
pub fn replace_sid(text: &str, placeholder: &str) -> String {
    let re = Regex::new(r"S[-–]1[-–]([0-9]+[-–])+[0-9]+").unwrap();
    re.replace_all(text, placeholder).to_string()
}

/// Replace GUID patterns with placeholder
pub fn replace_guid(text: &str, placeholder: &str) -> String {
    let re = Regex::new(r"\{?[0-9A-Fa-f]{8}[-–]([0-9A-Fa-f]{4}[-–]){3}[0-9A-Fa-f]{12}\}?").unwrap();
    re.replace_all(text, placeholder).to_string()
}

/// Replace user path patterns with placeholder
pub fn replace_user(text: &str, placeholder: &str) -> String {
    let re = Regex::new(r"(?i)(C:\\Users)\\[^\\]*\\").unwrap();
    re.replace_all(text, format!(r"$1\{}\", placeholder))
        .to_string()
}

/// Replace hash patterns with placeholder
pub fn replace_hash(text: &str, placeholder: &str) -> String {
    let re = Regex::new(r"\b(?:[A-Fa-f0-9]{64}|[A-Fa-f0-9]{40}|[A-Fa-f0-9]{32}|[A-Fa-f0-9]{20})\b").unwrap();
    re.replace_all(text, placeholder).to_string()
}

/// Replace filepath patterns with placeholder
pub fn replace_filepath(text: &str, placeholder: &str) -> String {
    // Simplified version of the complex filepath regex from Python
    let re = Regex::new(
        r#"(?x)
        (?P<opening>
            \b(?P<montage>[a-zA-Z]:[/\\])
            |[/\\][/\\](?<!http://)(?<!https://)
            |%\w+%[/\\]
        )
        (?:[^/\\<>:"|?\n\r ,'][^/\\<>:"|?\n\r]*[/\\])*
        (?:[^/\\<>:"'|?\n\r;, ][^/\\<>:"|?\n\r;, .]*)?
        (?:\.\w+)*
        "#
    ).unwrap();
    
    re.replace_all(text, format!("$opening{}", placeholder))
        .to_string()
}

/// Apply all placeholder transformations
pub fn apply_all_placeholders(text: &str, apply_filepath: bool) -> String {
    let mut result = text.to_string();
    result = replace_sid(&result, "<SID>");
    result = replace_guid(&result, "<GUID>");
    result = replace_hash(&result, "<HASH>");
    result = replace_user(&result, "<USER>");
    
    if apply_filepath {
        result = replace_filepath(&result, "<FILEPATH>");
    }
    
    result
}

/// Load data from CSV file
pub fn load_csv(path: &str, column_name: &str) -> Result<Vec<String>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;

    let mut data = Vec::new();
    
    for result in reader.deserialize() {
        let record: DataRow = result?;
        if let Some(value) = record.fields.get(column_name) {
            data.push(value.clone());
        }
    }

    Ok(data)
}

/// Load data from CSV with all columns
pub fn load_csv_with_columns(path: &str) -> Result<Vec<AHashMap<String, String>>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;

    let mut data = Vec::new();
    
    for result in reader.deserialize() {
        let record: DataRow = result?;
        data.push(record.fields);
    }

    Ok(data)
}

/// Load data from TXT file
pub fn load_txt(path: &str) -> Result<String> {
    Ok(std::fs::read_to_string(path)?)
}

/// Process dataframe-like data with slicing options
pub fn process_data(
    data: Vec<String>,
    n_lines: Option<usize>,
    percentage: Option<f64>,
    from_end: bool,
    randomize: bool,
) -> Vec<String> {
    let mut result = data;

    if randomize {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        result.shuffle(&mut rng);
    }

    if let Some(n) = n_lines {
        if from_end && result.len() > n {
            let skip_amount = result.len() - n;  // Calculate before moving
            result = result.into_iter().skip(skip_amount).collect();
        } else {
            result.truncate(n);
        }
    } else if let Some(pct) = percentage {
        let count = ((result.len() as f64 * pct) / 100.0) as usize;
        if from_end && result.len() > count {
            let skip_amount = result.len() - count;  // Calculate before moving
            result = result.into_iter().skip(skip_amount).collect();
        } else {
            result.truncate(count);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replace_sid() {
        let text = "my_S-1-0-0";
        assert_eq!(replace_sid(text, "<SID>"), "my_<SID>");

        let text2 = "other SID here: S-1-5-21-4242424242-424242-4242442-4242 should be replaced";
        assert_eq!(
            replace_sid(text2, "<placeholder>"),
            "other SID here: <placeholder> should be replaced"
        );
    }

    #[test]
    fn test_replace_user() {
        let text = r"C:\Users\some_user\some_folder";
        assert_eq!(
            replace_user(text, "<USER>"),
            r"C:\Users\<USER>\some_folder"
        );
    }

    #[test]
    fn test_replace_guid() {
        let text = r"Here is some {12345678-1234-1234-1234-123456789012}";
        assert_eq!(replace_guid(text, "<GUID>"), r"Here is some <GUID>");
    }

    #[test]
    fn test_replace_hash() {
        // SHA256
        let text = r"Here is some e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assert_eq!(replace_hash(text, "<HASH>"), r"Here is some <HASH>");

        // MD5
        let text2 = r"Here is some d41d8cd98f00b204e9800998ecf8427e";
        assert_eq!(replace_hash(text2, "<HASH>"), r"Here is some <HASH>");
    }

    #[test]
    fn test_csv_with_ahashmap_serde() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create temp CSV file
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "col1,col2").unwrap();
        writeln!(temp_file, "value1,value2").unwrap();
        writeln!(temp_file, "value3,value4").unwrap();
        temp_file.flush().unwrap();

        // Load CSV directly into AHashMap (tests serde feature)
        let result = load_csv_with_columns(temp_file.path().to_str().unwrap()).unwrap();
        
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].get("col1"), Some(&"value1".to_string()));
        assert_eq!(result[0].get("col2"), Some(&"value2".to_string()));
        assert_eq!(result[1].get("col1"), Some(&"value3".to_string()));
        assert_eq!(result[1].get("col2"), Some(&"value4".to_string()));
        
        // Verify it's actually an AHashMap (type check at compile time)
        let _: &AHashMap<String, String> = &result[0];
    }

    #[test]
    fn test_process_data_from_end() {
        let data = vec![
            "line1".to_string(),
            "line2".to_string(),
            "line3".to_string(),
            "line4".to_string(),
            "line5".to_string(),
        ];

        // Test taking last 2 lines
        let result = process_data(data.clone(), Some(2), None, true, false);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "line4");
        assert_eq!(result[1], "line5");

        // Test taking last 50%
        let result = process_data(data, None, Some(40.0), true, false);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "line4");
        assert_eq!(result[1], "line5");
    }
}